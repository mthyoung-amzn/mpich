/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

/* INSTRUMENTATION-ONLY module. See ofi_rndv_stats.h. Safe to remove. */

#include "mpidimpl.h"
#include "ofi_impl.h"
#include "ofi_rndv_stats.h"

MPIDI_OFI_rndv_stats_t MPIDI_OFI_rndv_stats;

static const char *const proto_name[MPIDI_OFI_STAT_NUM_PROTO] = {
    "eager",
    "inject",
    "rndv-read",
    "rndv-write",
    "rndv-pipeline",
    "rndv-direct",
};

/* per-rank report -- printed at finalize. We deliberately do NOT do a
 * collective reduce here: finalize tears down comms/endpoints and a collective
 * this late is fragile. Each rank prints its own line to stderr; aggregate
 * across ranks in post-processing (grep the tag). Rank is taken from the
 * process table, valid throughout finalize. */
void MPIDI_OFI_stats_report(void)
{
    if (!MPIDI_OFI_rndv_stats.enabled)
        return;

    MPIDI_OFI_rndv_stats_t *s = &MPIDI_OFI_rndv_stats;
    int rank = MPIR_Process.rank;

    /* one grep-able tag per line so a run can be aggregated with
     * `grep OFI_RNDV_STATS <log> | ...` */
    for (int i = 0; i < MPIDI_OFI_STAT_NUM_PROTO; i++) {
        if (s->msg_count[i] == 0)
            continue;
        fprintf(stderr,
                "[OFI_RNDV_STATS] rank=%d proto=%-13s msgs=%llu bytes=%llu\n",
                rank, proto_name[i], s->msg_count[i], s->byte_count[i]);
    }

    if (s->read_n > 0) {
        double ctl_us = (s->read_ctl_wait / (double) s->read_n) * 1e6;
        double bulk_us = (s->read_bulk / (double) s->read_n) * 1e6;
        double chunks = (double) s->read_chunks / (double) s->read_n;
        fprintf(stderr,
                "[OFI_RNDV_STATS] rank=%d read-timing n=%llu "
                "ctl_wait_us/msg=%.3f bulk_us/msg=%.3f chunks/msg=%.2f "
                "ctl_frac=%.3f\n",
                rank, s->read_n, ctl_us, bulk_us, chunks,
                (s->read_ctl_wait + s->read_bulk) > 0 ?
                s->read_ctl_wait / (s->read_ctl_wait + s->read_bulk) : 0.0);
    }

    if (s->mrreg_n > 0) {
        fprintf(stderr,
                "[OFI_RNDV_STATS] rank=%d mr-reg n=%llu reg_us/msg=%.3f close_us/msg=%.3f\n",
                rank, s->mrreg_n,
                (s->mrreg_time / (double) s->mrreg_n) * 1e6,
                (s->mrreg_close_time / (double) s->mrreg_n) * 1e6);
    }
    fflush(stderr);
}
