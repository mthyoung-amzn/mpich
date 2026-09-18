/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

/* -------------------------------------------------------------------------
 * OFI send-protocol instrumentation (INSTRUMENTATION-ONLY, safe to remove).
 *
 * Purpose: answer, at runtime, three questions for a given CVAR configuration:
 *   1. Which send protocol does each message actually fall into? (counts)
 *   2. In the rendezvous path, how much wall-clock time is spent
 *      registering memory vs. waiting on control messages (RTS/CTS +
 *      rdma_info) vs. moving bulk data?
 *   3. How many bytes / chunks each protocol moved.
 *
 * Enable at runtime with MPICH_OFI_RNDV_STATS=1. When disabled, every hook
 * below is a couple of predictable branches on a cached flag, so it is cheap
 * enough to leave compiled in; set the env only when measuring.
 *
 * Timing model (receiver side, RDMA-read protocol):
 *
 *   RTS recv (recv_rndv_event)  ── t_rts
 *        │  post recv-for-hdr + send CTS        ┐ control wait
 *        ▼                                      │ (t_first_read - t_rts)
 *   rdma_info recv (recv_mrs_event)             ┘
 *        │  compute chunks
 *        ▼
 *   first fi_read issued        ── t_first_read ┐ bulk transfer
 *        │  ... reads ...                       │ (t_complete - t_first_read)
 *        ▼                                      ┘
 *   last chunk complete / ack   ── t_complete
 *
 * Sender side, RDMA-read protocol: we time the fi_mr_reg block in
 * prepare_rdma_info (this is the "how much time registering memory" number).
 * ------------------------------------------------------------------------- */

#ifndef OFI_RNDV_STATS_H_INCLUDED
#define OFI_RNDV_STATS_H_INCLUDED

#include "mpl.h"

/* Protocol buckets. Kept in sync with the array of names below. */
enum MPIDI_OFI_stat_proto {
    MPIDI_OFI_STAT_EAGER = 0,   /* fi_tsend/fi_tsenddata below EAGER_THRESH */
    MPIDI_OFI_STAT_INJECT,      /* fi_tinject lightweight path */
    MPIDI_OFI_STAT_RNDV_READ,
    MPIDI_OFI_STAT_RNDV_WRITE,
    MPIDI_OFI_STAT_RNDV_PIPELINE,
    MPIDI_OFI_STAT_RNDV_DIRECT,
    MPIDI_OFI_STAT_NUM_PROTO
};

typedef struct MPIDI_OFI_rndv_stats {
    int enabled;                /* cached MPICH_OFI_RNDV_STATS */

    /* per-protocol message + byte counts (counted where the protocol is
     * selected: eager/inject on the sender, rndv on the receiver's CTS so a
     * message is counted exactly once) */
    unsigned long long msg_count[MPIDI_OFI_STAT_NUM_PROTO];
    unsigned long long byte_count[MPIDI_OFI_STAT_NUM_PROTO];

    /* rndv-read timing accumulators (receiver side) */
    unsigned long long read_n;          /* messages timed */
    double read_ctl_wait;               /* sum(t_first_read - t_rts): RTS/CTS + rdma_info wait */
    double read_bulk;                   /* sum(t_complete - t_first_read): actual reads */
    unsigned long long read_chunks;     /* total fi_reads issued */

    /* sender-side MR registration timing (rndv read+write) */
    unsigned long long mrreg_n;         /* number of prepare_rdma_info calls */
    double mrreg_time;                  /* sum wall-clock in the fi_mr_reg loop */
    double mrreg_close_time;            /* sum wall-clock in the fi_close loop */
} MPIDI_OFI_rndv_stats_t;

extern MPIDI_OFI_rndv_stats_t MPIDI_OFI_rndv_stats;

MPL_STATIC_INLINE_PREFIX void MPIDI_OFI_stats_init(void)
{
    memset(&MPIDI_OFI_rndv_stats, 0, sizeof(MPIDI_OFI_rndv_stats));
    char *e = getenv("MPICH_OFI_RNDV_STATS");
    MPIDI_OFI_rndv_stats.enabled = (e && *e && *e != '0') ? 1 : 0;
}

MPL_STATIC_INLINE_PREFIX int MPIDI_OFI_stats_on(void)
{
    return MPIDI_OFI_rndv_stats.enabled;
}

/* wall clock in seconds; only called when stats are on */
MPL_STATIC_INLINE_PREFIX double MPIDI_OFI_stats_now(void)
{
    MPL_time_t t;
    double sec;
    MPL_wtime(&t);
    MPL_wtime_todouble(&t, &sec);
    return sec;
}

MPL_STATIC_INLINE_PREFIX void MPIDI_OFI_stats_count(enum MPIDI_OFI_stat_proto proto,
                                                    MPI_Aint bytes)
{
    if (!MPIDI_OFI_rndv_stats.enabled)
        return;
    MPIDI_OFI_rndv_stats.msg_count[proto]++;
    MPIDI_OFI_rndv_stats.byte_count[proto] += (unsigned long long) bytes;
}

/* Print an aggregated summary. Called from finalize on every rank; the reduce
 * + rank-0 print is done inside so callers stay simple. */
void MPIDI_OFI_stats_report(void);

#endif /* OFI_RNDV_STATS_H_INCLUDED */
