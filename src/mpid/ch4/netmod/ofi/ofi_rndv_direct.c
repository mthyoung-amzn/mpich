/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpidimpl.h"
#include "ofi_impl.h"
#include "ofi_rndv.h"

/* Striped "direct" rendezvous data transfer.
 *
 * Unlike the pipeline path (which chunks small and keeps many chunks in flight to
 * overlap the device<->host staging copy), direct has no copy to hide: it sends
 * straight from the user buffer. So it uses the LARGEST possible slice per NIC --
 * exactly num_nics tagged sends/recvs, buf_sz/num_nics bytes each -- which minimizes
 * WQE count and maximizes MR-registration amortization. There is no inflight window
 * and no async poll loop; completion is a simple count of num_nics chunks home.
 *
 * NIC i is paired with peer NIC i (see MPIDI_OFI_av_to_phys(av, .., nic, .., nic)),
 * and both sender and receiver derive the identical even split from data_sz and the
 * globally-agreed num_nics, so no per-message negotiation of offsets is needed.
 *
 * Restrictions (enforced by the caller in ofi_rndv.c): contiguous datatype only, and
 * data_sz <= max_msg_size (each slice is necessarily <= data_sz, so within limits).
 */

/* per-chunk OFI request, one per NIC; freed in the chunk-completion event */
struct direct_chunk_req {
    char pad[MPIDI_REQUEST_HDR_SIZE];
    struct fi_context context[MPIDI_OFI_CONTEXT_STRUCTS];
    int event_id;
    MPIR_Request *req;          /* parent sreq or rreq */
};

/* even split: NIC i owns [i*slice, i*slice + slice_sz(i)); last NIC absorbs the
 * remainder so the slices always sum to exactly data_sz. */
static inline MPI_Aint direct_slice_base(MPI_Aint data_sz, int num_nics, int nic)
{
    return (data_sz / num_nics) * nic;
}

static inline MPI_Aint direct_slice_sz(MPI_Aint data_sz, int num_nics, int nic)
{
    MPI_Aint slice = data_sz / num_nics;
    if (nic == num_nics - 1) {
        return data_sz - slice * (num_nics - 1);
    }
    return slice;
}

/* decide whether this contiguous buffer should be sent as a registered device MR
 * (GDR) -- mirrors the need_mr logic in MPIDI_OFI_send_normal / ofi_send.h. */
static inline bool direct_need_mr(MPL_pointer_attr_t * attr)
{
    return MPIDI_OFI_ENABLE_HMEM && MPIDI_OFI_ENABLE_MR_HMEM &&
        MPL_gpu_attr_is_strict_dev(attr);
}

/* ---- sender side ---- */

int MPIDI_OFI_direct_send(MPIR_Request * sreq, int tag)
{
    int mpi_errno = MPI_SUCCESS;
    /* Read the common fields (populated by the rndv arm) at issue time; store only
     * the survive-to-completion striping state (chunks_remain, mrs, datatype) into
     * the direct variant. The two views overlay the same request bytes, so the
     * common fields must be consumed BEFORE we write the direct variant. */
    MPIDI_OFI_rndv_common_t *c = &MPIDI_OFI_AMREQ_COMMON(sreq);

    MPIR_Assert(!c->need_pack);

    int num_nics = MPIDI_OFI_global.num_nics;
    MPI_Aint data_sz = MPL_MIN(c->remote_data_sz, c->data_sz);
    int vci_local = c->vci_local;
    int vci_remote = c->vci_remote;
    struct MPIDI_av_entry *av = c->av;
    uint64_t match_bits = c->match_bits;
    MPL_pointer_attr_t attr = c->attr;
    MPI_Datatype datatype = c->datatype;
    const void *buf = c->buf;

    MPIR_FUNC_ENTER;

    MPI_Aint true_extent, true_lb;
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    const char *base = MPIR_get_contig_ptr(buf, true_lb);

    bool need_mr = direct_need_mr(&attr);

    /* now switch to the base (direct) request view for survive-to-completion state */
    MPIDI_OFI_REQUEST(sreq, datatype) = datatype;
    MPIDI_OFI_REQUEST(sreq, chunks_remain) = num_nics;
    MPIDI_OFI_REQUEST(sreq, mrs) = NULL;
    struct fid_mr **mrs = NULL;
    if (need_mr) {
        mrs = MPL_malloc(num_nics * sizeof(struct fid_mr *), MPL_MEM_OTHER);
        MPIR_Assertp(mrs);
        for (int i = 0; i < num_nics; i++) {
            mrs[i] = NULL;
        }
        MPIDI_OFI_REQUEST(sreq, mrs) = mrs;
    }

    for (int nic = 0; nic < num_nics; nic++) {
        MPI_Aint off = direct_slice_base(data_sz, num_nics, nic);
        MPI_Aint slice_sz = direct_slice_sz(data_sz, num_nics, nic);
        const char *slice = base + off;

        int ctx_idx = MPIDI_OFI_get_ctx_index(vci_local, nic);
        fi_addr_t addr = MPIDI_OFI_av_to_phys(av, vci_local, nic, vci_remote, nic);

        void *desc = NULL;
        if (need_mr && slice_sz > 0) {
            struct fid_mr *mr = NULL;
            mpi_errno = MPIDI_OFI_register_memory_and_bind((char *) slice, slice_sz,
                                                           &attr, ctx_idx, &mr);
            MPIR_ERR_CHECK(mpi_errno);
            if (mr != NULL) {
                mrs[nic] = mr;
                desc = fi_mr_desc(mr);
            }
        }

        struct direct_chunk_req *creq =
            MPL_malloc(sizeof(struct direct_chunk_req), MPL_MEM_BUFFER);
        MPIR_Assertp(creq);
        creq->event_id = MPIDI_OFI_EVENT_DIRECT_SEND_CHUNK;
        creq->req = sreq;

        MPIDI_OFI_CALL_RETRY(fi_tsend(MPIDI_OFI_global.ctx[ctx_idx].tx,
                                      slice, slice_sz, desc, addr, match_bits,
                                      (void *) &creq->context), vci_local, tsend);
        MPIR_T_PVAR_COUNTER_INC(MULTINIC, nic_sent_bytes_count[nic], slice_sz);
    }

  fn_exit:
    MPIR_FUNC_EXIT;
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIDI_OFI_direct_send_chunk_event(struct fi_cq_tagged_entry *wc, MPIR_Request * r)
{
    int mpi_errno = MPI_SUCCESS;
    struct direct_chunk_req *creq = (void *) r;
    MPIR_Request *sreq = creq->req;

    MPL_free(creq);

    if (--MPIDI_OFI_REQUEST(sreq, chunks_remain) == 0) {
        struct fid_mr **mrs = MPIDI_OFI_REQUEST(sreq, mrs);
        if (mrs) {
            for (int i = 0; i < MPIDI_OFI_global.num_nics; i++) {
                if (mrs[i]) {
                    mpi_errno = MPIDI_OFI_mr_complete(mrs[i], true);
                    MPIR_ERR_CHECK(mpi_errno);
                }
            }
            MPL_free(mrs);
            MPIDI_OFI_REQUEST(sreq, mrs) = NULL;
        }
        MPIR_Datatype_release_if_not_builtin(MPIDI_OFI_REQUEST(sreq, datatype));
        MPIDI_Request_complete_fast(sreq);
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* ---- receiver side ---- */

int MPIDI_OFI_direct_recv(MPIR_Request * rreq, int tag, int vci_src, int vci_dst)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_OFI_rndv_common_t *c = &MPIDI_OFI_AMREQ_COMMON(rreq);

    MPIR_Assert(!c->need_pack);

    int num_nics = MPIDI_OFI_global.num_nics;
    /* receiver knows the transfer length from the RTS/CTS exchange; use the smaller
     * of posted and remote size, matching the sender's data_sz. */
    MPI_Aint data_sz = c->data_sz;
    if (c->remote_data_sz != -1) {
        data_sz = MPL_MIN(c->remote_data_sz, c->data_sz);
    }
    int vci_local = c->vci_local;
    int vci_remote = c->vci_remote;
    struct MPIDI_av_entry *av = c->av;
    uint64_t match_bits = c->match_bits;
    MPL_pointer_attr_t attr = c->attr;
    MPI_Datatype datatype = c->datatype;
    void *buf = (void *) c->buf;

    MPIR_FUNC_ENTER;

    MPI_Aint true_extent, true_lb;
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    char *base = MPIR_get_contig_ptr(buf, true_lb);

    bool need_mr = direct_need_mr(&attr);

    MPIDI_OFI_REQUEST(rreq, datatype) = datatype;
    MPIDI_OFI_REQUEST(rreq, chunks_remain) = num_nics;
    MPIDI_OFI_REQUEST(rreq, mrs) = NULL;
    struct fid_mr **mrs = NULL;
    if (need_mr) {
        mrs = MPL_malloc(num_nics * sizeof(struct fid_mr *), MPL_MEM_OTHER);
        MPIR_Assertp(mrs);
        for (int i = 0; i < num_nics; i++) {
            mrs[i] = NULL;
        }
        MPIDI_OFI_REQUEST(rreq, mrs) = mrs;
    }

    for (int nic = 0; nic < num_nics; nic++) {
        MPI_Aint off = direct_slice_base(data_sz, num_nics, nic);
        MPI_Aint slice_sz = direct_slice_sz(data_sz, num_nics, nic);
        char *slice = base + off;

        int ctx_idx = MPIDI_OFI_get_ctx_index(vci_local, nic);
        fi_addr_t addr = MPIDI_OFI_av_to_phys(av, vci_local, nic, vci_remote, nic);

        void *desc = NULL;
        if (need_mr && slice_sz > 0) {
            struct fid_mr *mr = NULL;
            mpi_errno = MPIDI_OFI_register_memory_and_bind(slice, slice_sz,
                                                           &attr, ctx_idx, &mr);
            MPIR_ERR_CHECK(mpi_errno);
            if (mr != NULL) {
                mrs[nic] = mr;
                desc = fi_mr_desc(mr);
            }
        }

        struct direct_chunk_req *creq =
            MPL_malloc(sizeof(struct direct_chunk_req), MPL_MEM_BUFFER);
        MPIR_Assertp(creq);
        creq->event_id = MPIDI_OFI_EVENT_DIRECT_RECV_CHUNK;
        creq->req = rreq;

        MPIDI_OFI_CALL_RETRY(fi_trecv(MPIDI_OFI_global.ctx[ctx_idx].rx,
                                      slice, slice_sz, desc, addr, match_bits, 0ULL,
                                      (void *) &creq->context), vci_local, trecv);
    }

  fn_exit:
    MPIR_FUNC_EXIT;
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIDI_OFI_direct_recv_chunk_event(struct fi_cq_tagged_entry *wc, MPIR_Request * r)
{
    int mpi_errno = MPI_SUCCESS;
    struct direct_chunk_req *creq = (void *) r;
    MPIR_Request *rreq = creq->req;

    MPL_free(creq);

    if (--MPIDI_OFI_REQUEST(rreq, chunks_remain) == 0) {
        struct fid_mr **mrs = MPIDI_OFI_REQUEST(rreq, mrs);
        if (mrs) {
            for (int i = 0; i < MPIDI_OFI_global.num_nics; i++) {
                if (mrs[i]) {
                    mpi_errno = MPIDI_OFI_mr_complete(mrs[i], true);
                    MPIR_ERR_CHECK(mpi_errno);
                }
            }
            MPL_free(mrs);
            MPIDI_OFI_REQUEST(rreq, mrs) = NULL;
        }
        MPIR_Datatype_release_if_not_builtin(MPIDI_OFI_REQUEST(rreq, datatype));
        MPIDI_Request_complete_fast(rreq);
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
