/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "mpidimpl.h"
#include "ofi_impl.h"
#include "ofi_rndv.h"

/* The "stripe" rendezvous protocol: a send-based multi-rail data transfer.
 *
 * Unlike pipeline (which chunks small and keeps many chunks in flight to overlap the
 * device<->host staging copy), stripe has no copy to hide: it sends straight from the
 * user buffer. So it uses the LARGEST possible slice per NIC -- exactly num_nics
 * tagged sends/recvs, buf_sz/num_nics bytes each -- which minimizes WQE count and
 * maximizes MR-registration amortization. There is no inflight window and no async
 * poll loop; completion is a simple count of num_nics chunks home.
 *
 * Unlike read/write, stripe is a two-sided tagged send: no rkey exchange, no
 * rdma_info header, no remote-side registration -- the receiver just posts matching
 * recvs. It is the pipeline path minus the host bounce and the genq pool.
 *
 * NIC i is paired with peer NIC i (see MPIDI_OFI_av_to_phys(av, .., nic, .., nic)),
 * and both sender and receiver derive the identical even split from data_sz and the
 * globally-agreed num_nics, so no per-message negotiation of offsets is needed.
 *
 * Restrictions (enforced by the caller in ofi_rndv.c): contiguous datatype only, and
 * data_sz <= max_msg_size (each slice is necessarily <= data_sz, so within limits).
 */

/* per-chunk OFI request, one per NIC; freed in the chunk-completion event */
struct stripe_chunk_req {
    char pad[MPIDI_REQUEST_HDR_SIZE];
    struct fi_context context[MPIDI_OFI_CONTEXT_STRUCTS];
    int event_id;
    MPIR_Request *req;          /* parent sreq or rreq */
};

/* even split: NIC i owns [i*slice, i*slice + slice_sz(i)); last NIC absorbs the
 * remainder so the slices always sum to exactly data_sz. */
static inline MPI_Aint stripe_slice_base(MPI_Aint data_sz, int num_nics, int nic)
{
    return (data_sz / num_nics) * nic;
}

static inline MPI_Aint stripe_slice_sz(MPI_Aint data_sz, int num_nics, int nic)
{
    MPI_Aint slice = data_sz / num_nics;
    if (nic == num_nics - 1) {
        return data_sz - slice * (num_nics - 1);
    }
    return slice;
}

/* decide whether this contiguous buffer should be sent as a registered device MR
 * (GDR) -- mirrors the need_mr logic in MPIDI_OFI_send_normal / ofi_send.h. */
static inline bool stripe_need_mr(MPL_pointer_attr_t * attr)
{
    return MPIDI_OFI_ENABLE_HMEM && MPIDI_OFI_ENABLE_MR_HMEM &&
        MPL_gpu_attr_is_strict_dev(attr);
}

/* ---- sender side ---- */

/* Phase 2: the receiver has posted its num_nics slice recvs and sent back the
 * "posted" ack; now fire the slices. Called from the posted-ack event. Splitting
 * issue into two phases (send_sz hdr, then wait for the posted ack, then slices)
 * costs one extra round trip but guarantees every slice recv is posted before its
 * send -- zero unexpected chunks. The RTT is why stripe is size-gated: only chosen
 * for messages large enough that the transfer dwarfs the extra RTT. */
static int stripe_send_slices(MPIR_Request * sreq)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_OFI_rndv_common_t *c = &MPIDI_OFI_AMREQ_COMMON(sreq);

    int num_nics = MPIDI_OFI_global.num_nics;
    MPI_Aint data_sz = MPL_MIN(c->remote_data_sz, c->data_sz);
    int vci_local = c->vci_local;
    int vci_remote = c->vci_remote;
    struct MPIDI_av_entry *av = c->av;
    uint64_t match_bits = c->match_bits;
    MPL_pointer_attr_t attr = c->attr;
    MPI_Datatype datatype = c->datatype;
    const void *buf = c->buf;

    MPI_Aint true_extent, true_lb;
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    const char *base = MPIR_get_contig_ptr(buf, true_lb);

    bool need_mr = stripe_need_mr(&attr);

    /* now switch to the stripe request view for survive-to-completion state */
    MPIDI_OFI_AMREQ_STRIPE(sreq).datatype = datatype;
    MPIDI_OFI_AMREQ_STRIPE(sreq).chunks_remain = num_nics;
    MPIDI_OFI_AMREQ_STRIPE(sreq).mrs = NULL;
    struct fid_mr **mrs = NULL;
    if (need_mr) {
        mrs = MPL_malloc(num_nics * sizeof(struct fid_mr *), MPL_MEM_OTHER);
        MPIR_Assertp(mrs);
        for (int i = 0; i < num_nics; i++) {
            mrs[i] = NULL;
        }
        MPIDI_OFI_AMREQ_STRIPE(sreq).mrs = mrs;
    }

    for (int nic = 0; nic < num_nics; nic++) {
        MPI_Aint off = stripe_slice_base(data_sz, num_nics, nic);
        MPI_Aint slice_sz = stripe_slice_sz(data_sz, num_nics, nic);
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

        struct stripe_chunk_req *creq =
            MPL_malloc(sizeof(struct stripe_chunk_req), MPL_MEM_BUFFER);
        MPIR_Assertp(creq);
        creq->event_id = MPIDI_OFI_EVENT_STRIPE_SEND_CHUNK;
        creq->req = sreq;

        MPIDI_OFI_CALL_RETRY(fi_tsend(MPIDI_OFI_global.ctx[ctx_idx].tx,
                                      slice, slice_sz, desc, addr, match_bits,
                                      (void *) &creq->context), vci_local, tsend);
        MPIR_T_PVAR_COUNTER_INC(MULTINIC, nic_sent_bytes_count[nic], slice_sz);
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* Phase 1 (runs off CTS): send our transfer length to the receiver, then post a
 * recv for the receiver's "posted" ack. We do NOT send slices here -- we wait for
 * the ack so the receiver's slice recvs are guaranteed posted first. */
int MPIDI_OFI_stripe_send(MPIR_Request * sreq, int tag)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_OFI_rndv_common_t *c = &MPIDI_OFI_AMREQ_COMMON(sreq);

    MPIR_Assert(!c->need_pack);

    MPIR_FUNC_ENTER;

    /* send our transfer length (send_sz) to the receiver. Unconditional -- stripe
     * never relies on the size riding the RTS cq_data, so it is uniform on every
     * provider (e.g. EFA, cq_data_size==4). */
    mpi_errno = MPIDI_OFI_RNDV_send_hdr(&c->data_sz, sizeof(MPI_Aint),
                                        c->av, c->vci_local, c->vci_remote, c->match_bits);
    MPIR_ERR_CHECK(mpi_errno);

    /* post the recv for the receiver's "posted" ack; its event fires the slices.
     * The ack body is empty -- its arrival IS the signal that the receiver posted
     * its num_nics slice recvs. */
    mpi_errno = MPIDI_OFI_RNDV_recv_hdr(sreq, MPIDI_OFI_EVENT_STRIPE_SEND_POSTED_ACK,
                                        0, c->av, c->vci_local, c->vci_remote, c->match_bits);
    MPIR_ERR_CHECK(mpi_errno);

  fn_exit:
    MPIR_FUNC_EXIT;
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* the receiver's "posted" ack landed: its slice recvs are up, fire the slices. */
int MPIDI_OFI_stripe_send_posted_ack_event(struct fi_cq_tagged_entry *wc, MPIR_Request * r)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_Request *sreq = MPIDI_OFI_RNDV_GET_CONTROL_REQ(r);

    MPL_free(r);

    mpi_errno = stripe_send_slices(sreq);
    MPIR_ERR_CHECK(mpi_errno);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIDI_OFI_stripe_send_chunk_event(struct fi_cq_tagged_entry *wc, MPIR_Request * r)
{
    int mpi_errno = MPI_SUCCESS;
    struct stripe_chunk_req *creq = (void *) r;
    MPIR_Request *sreq = creq->req;

    MPL_free(creq);

    if (--MPIDI_OFI_AMREQ_STRIPE(sreq).chunks_remain == 0) {
        struct fid_mr **mrs = MPIDI_OFI_AMREQ_STRIPE(sreq).mrs;
        if (mrs) {
            for (int i = 0; i < MPIDI_OFI_global.num_nics; i++) {
                if (mrs[i]) {
                    mpi_errno = MPIDI_OFI_mr_complete(mrs[i], true);
                    MPIR_ERR_CHECK(mpi_errno);
                }
            }
            MPL_free(mrs);
            MPIDI_OFI_AMREQ_STRIPE(sreq).mrs = NULL;
        }
        MPIR_Datatype_release_if_not_builtin(MPIDI_OFI_AMREQ_STRIPE(sreq).datatype);
        MPIDI_Request_complete_fast(sreq);
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* ---- receiver side ---- */

/* Post the num_nics matching recvs once the transfer length is known. Called from
 * the datasize event (never inline) because stripe always waits for the length
 * handshake before it can compute the even split. */
static int stripe_post_recvs(MPIR_Request * rreq)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_OFI_rndv_common_t *c = &MPIDI_OFI_AMREQ_COMMON(rreq);

    int num_nics = MPIDI_OFI_global.num_nics;
    MPI_Aint data_sz = MPL_MIN(c->remote_data_sz, c->data_sz);
    int vci_local = c->vci_local;
    int vci_remote = c->vci_remote;
    struct MPIDI_av_entry *av = c->av;
    uint64_t match_bits = c->match_bits;
    MPL_pointer_attr_t attr = c->attr;
    MPI_Datatype datatype = c->datatype;
    void *buf = (void *) c->buf;

    MPI_Aint true_extent, true_lb;
    MPIR_Type_get_true_extent_impl(datatype, &true_lb, &true_extent);
    char *base = MPIR_get_contig_ptr(buf, true_lb);

    bool need_mr = stripe_need_mr(&attr);

    /* switch to the stripe view for survive-to-completion state */
    MPIDI_OFI_AMREQ_STRIPE(rreq).datatype = datatype;
    MPIDI_OFI_AMREQ_STRIPE(rreq).chunks_remain = num_nics;
    MPIDI_OFI_AMREQ_STRIPE(rreq).mrs = NULL;
    struct fid_mr **mrs = NULL;
    if (need_mr) {
        mrs = MPL_malloc(num_nics * sizeof(struct fid_mr *), MPL_MEM_OTHER);
        MPIR_Assertp(mrs);
        for (int i = 0; i < num_nics; i++) {
            mrs[i] = NULL;
        }
        MPIDI_OFI_AMREQ_STRIPE(rreq).mrs = mrs;
    }

    for (int nic = 0; nic < num_nics; nic++) {
        MPI_Aint off = stripe_slice_base(data_sz, num_nics, nic);
        MPI_Aint slice_sz = stripe_slice_sz(data_sz, num_nics, nic);
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

        struct stripe_chunk_req *creq =
            MPL_malloc(sizeof(struct stripe_chunk_req), MPL_MEM_BUFFER);
        MPIR_Assertp(creq);
        creq->event_id = MPIDI_OFI_EVENT_STRIPE_RECV_CHUNK;
        creq->req = rreq;

        MPIDI_OFI_CALL_RETRY(fi_trecv(MPIDI_OFI_global.ctx[ctx_idx].rx,
                                      slice, slice_sz, desc, addr, match_bits, 0ULL,
                                      (void *) &creq->context), vci_local, trecv);
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIDI_OFI_stripe_recv(MPIR_Request * rreq, int tag, int vci_src, int vci_dst)
{
    int mpi_errno = MPI_SUCCESS;
    MPIDI_OFI_rndv_common_t *c = &MPIDI_OFI_AMREQ_COMMON(rreq);

    MPIR_Assert(!c->need_pack);

    MPIR_FUNC_ENTER;

    /* Always wait for the sender's transfer length via the datasize handshake, then
     * post the matching recvs in the datasize event. stripe does not read the size
     * from the RTS cq_data at all -- the handshake is unconditional, so the slice
     * geometry is known the same way on every provider. */
    mpi_errno = MPIDI_OFI_RNDV_recv_hdr(rreq, MPIDI_OFI_EVENT_STRIPE_RECV_DATASIZE,
                                        sizeof(MPI_Aint), c->av, c->vci_local, c->vci_remote,
                                        c->match_bits);
    MPIR_ERR_CHECK(mpi_errno);

  fn_exit:
    MPIR_FUNC_EXIT;
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

/* datasize handshake landed: learn the transfer length, set the recv count, then
 * post the num_nics matching recvs. */
int MPIDI_OFI_stripe_recv_datasize_event(struct fi_cq_tagged_entry *wc, MPIR_Request * r)
{
    int mpi_errno = MPI_SUCCESS;
    MPIR_Request *rreq = MPIDI_OFI_RNDV_GET_CONTROL_REQ(r);
    MPIDI_OFI_rndv_common_t *c = &MPIDI_OFI_AMREQ_COMMON(rreq);

    MPI_Aint *hdr_data_sz = MPIDI_OFI_RNDV_GET_CONTROL_HDR(r);
    MPIDI_OFI_RNDV_update_count(rreq, *hdr_data_sz);
    c->remote_data_sz = *hdr_data_sz;

    MPL_free(r);

    /* post our num_nics slice recvs FIRST ... */
    mpi_errno = stripe_post_recvs(rreq);
    MPIR_ERR_CHECK(mpi_errno);

    /* ... then tell the sender we are posted (empty "posted" ack). Ordering matters:
     * the recvs are up before the sender can see this ack and start sending slices,
     * so no slice ever arrives unexpected. */
    mpi_errno = MPIDI_OFI_RNDV_send_hdr(NULL, 0, c->av, c->vci_local, c->vci_remote,
                                        c->match_bits);
    MPIR_ERR_CHECK(mpi_errno);

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}

int MPIDI_OFI_stripe_recv_chunk_event(struct fi_cq_tagged_entry *wc, MPIR_Request * r)
{
    int mpi_errno = MPI_SUCCESS;
    struct stripe_chunk_req *creq = (void *) r;
    MPIR_Request *rreq = creq->req;

    MPL_free(creq);

    if (--MPIDI_OFI_AMREQ_STRIPE(rreq).chunks_remain == 0) {
        struct fid_mr **mrs = MPIDI_OFI_AMREQ_STRIPE(rreq).mrs;
        if (mrs) {
            for (int i = 0; i < MPIDI_OFI_global.num_nics; i++) {
                if (mrs[i]) {
                    mpi_errno = MPIDI_OFI_mr_complete(mrs[i], true);
                    MPIR_ERR_CHECK(mpi_errno);
                }
            }
            MPL_free(mrs);
            MPIDI_OFI_AMREQ_STRIPE(rreq).mrs = NULL;
        }
        MPIR_Datatype_release_if_not_builtin(MPIDI_OFI_AMREQ_STRIPE(rreq).datatype);
        MPIDI_Request_complete_fast(rreq);
    }

  fn_exit:
    return mpi_errno;
  fn_fail:
    goto fn_exit;
}
