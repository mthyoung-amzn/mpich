# Persistent-Request MR Caching — Working Status / Handoff

Last updated: 2026-09-08

Workspace: `~/git/mpich` (branch `main`, in-source configured build). **No commits made** —
all changes are working-tree only.

---

## What we're building

Cache EFA/libfabric memory regions (MRs) on **persistent request objects** so that
repeated `MPI_Start` calls reuse a single registration instead of paying
`fi_mr_reg` + `fi_close` on every iteration. This bypasses MPICH's rudimentary global
MR cache for persistent requests.

**v1 scope: Regime A only — the GPU/HMEM direct send/recv path** (customer priority;
device MR registration via dmabuf is the most expensive). Regime B (host rendezvous
RDMA) and `MPI_ANY_SOURCE` are deferred.

Full design rationale: `doc/design/persistent-request-mr-caching.md`.

---

## Key facts established (verified against code + MPI-5.0 spec)

- Persistent pt2pt requests today just store args (`MPIDI_prequest_t`) and spawn a fresh
  ephemeral `real_request` on each start (`ch4_startall.h` → `MPID_Isend`/`Irecv`). No reuse.
- Two MR regimes in OFI:
  - **A (GPU/HMEM direct):** `MPIDI_OFI_register_memory_and_bind` → global MR cache
    (`mr_cache.c`). Only for device buffers, `MR_HMEM`, above `GPU_RDMA_THRESHOLD`.
  - **B (host rendezvous):** `prepare_rdma_info` registers per-NIC and `fi_close`s every
    transfer — no cache at all. (Deferred.)
- Global MR cache is small/evictable (default cap 16, O(n) search+LRU) — a long-lived
  persistent request can have its MR evicted mid-loop.
- Spec (MPI-5.0 §4.9): buffer addr/count/type are bound for the request's lifetime (only
  contents change) ⇒ the MR is invariant across starts. Free-only-when-inactive ⇒
  `MPI_Request_free` is the correct deregistration point. Spec rationale explicitly blesses
  binding internal resources (like MRs) to persistent requests.

---

## Design decisions (settled)

1. **Forward-thread an opaque, netmod-owned handle** (option "a") from `prequest_start`
   into the send/recv call. Avoids the race where a synchronous GDR-direct registration
   fires before a back-pointer could be set.
2. **Tagged base struct** `MPIDI_NM_persist_base_t { int owner; }` (shared, NM-agnostic),
   embedded as the **first member** of the OFI-private `MPIDI_OFI_persist_mr_t`. Downcasts
   are guarded on `owner` (reuses existing `MPIDI_NETMOD`/`MPIDI_SHM` enum from
   `ch4_types.h` — no new enum). Only one netmod is compiled in, so the only real
   heterogeneity is NM-vs-SHM.
3. **Carrier seam:** one new `persist_state` parameter on the `MPIDI_NM_mpi_isend`/`irecv`
   dispatch (added via `ch4_api.txt` + regenerated). Mirrors the existing `partner`
   precedent. `partner` and `persist_state` kept **separate** (different lifetimes; not
   logically mutually exclusive — a future anysource impl could need both).
4. **Anysource opted out** in v1 by leaving the handle NULL for `MPI_ANY_SOURCE` — a
   product decision, not a logical constraint. No assertion coupling the two params (would
   be a landmine when anysource lands).
5. **CVAR** `MPIR_CVAR_CH4_OFI_PERSISTENT_MR` (bool, default true) as toggle/kill-switch.

---

## Implementation (done — builds clean, EXIT=0, no warnings in our files)

Files changed:

- `src/mpid/ch4/include/mpidpre.h` — `MPIDI_NM_persist_base_t` (before `netmodpre.h`);
  `nm_persist` field on `MPIDI_prequest_t`; forward-decl of `MPIDI_NM_prequest_free_hook`
  (placed early to fix an include-order implicit-declaration warning).
- `src/mpid/ch4/netmod/ofi/ofi_pre.h` — `MPIDI_OFI_persist_mr_t` (base first member) +
  `MPIDI_OFI_persist_kind` enum.
- `src/mpid/ch4/netmod/ofi/mr_cache.c` — CVAR; `MPIDI_NM_persist_alloc`,
  `MPIDI_OFI_persist_get_or_reg_mr` (uses GPU buffer bounds; reuse if base/len/ctx match,
  else fi_close stale + register into slot, bypassing global cache; falls back to
  `register_memory_and_bind` if CVAR off or not owned), `MPIDI_NM_prequest_free_hook`.
- `src/mpid/ch4/netmod/ofi/ofi_impl.h` — prototypes + `MPIDI_OFI_PERSIST_OWNS(ps)` macro.
- `src/mpid/ch4/ch4_api.txt` — added `persist_state` param to `mpi_isend`/`mpi_irecv`
  (NM+SHM). Regenerated with `maint/gen_ch4_api.py` (run inside `autogen.sh`).
- `src/mpid/ch4/src/ch4_send.h`, `ch4_recv.h` — thread `persist_state` through
  `MPIDI_isend`/`irecv`; non-persistent callers and anysource pass NULL.
- `src/mpid/ch4/src/ch4_startall.h` — lazy-alloc handle on first start (skip anysource +
  self-comm); forward via `MPIDI_isend`/`irecv`.
- `src/mpid/ch4/netmod/ofi/ofi_send.h`, `ofi_recv.h` — consume handle at the HMEM
  registration sites; skip stashing the MR on the request (and skip completion-time
  `mr_complete`) when persistent-owned.
- `src/mpid/ch4/src/ch4_request.h` — `MPID_Prequest_free_hook` calls
  `MPIDI_NM_prequest_free_hook` (guarded `#ifdef HAVE_CH4_NETMOD_OFI`).
- UCX (`ucx_send.h`/`ucx_recv.h`), AM-fallback, and SHM (`shm_p2p.h`,
  `shm_am_fallback_*`) signatures updated to match regenerated dispatch (param ignored).
- `src/mpid/ch4/include/mpidch4.h` — reordered includes (`ch4_recv.h` before
  `ch4_startall.h`).

Build issues fixed: include-order for `MPIDI_irecv`; `MPL_USED` misuse in the free hook;
`MPIDI_NM_prequest_free_hook` implicit-declaration (declared early in `mpidpre.h`).

---

## Testing so far

- `test/persist_mr_bench.c` — GPU-aware (`-DUSE_CUDA`, host fallback) persistent-vs-
  nonpersistent ping-pong with per-iteration correctness check.
- Local 2-rank run (host buffers, no GPU on dev host): **CORRECT** with CVAR on AND off,
  no crash, free hook exercised. Timings identical on host — expected, since without a GPU
  `need_mr`/`register_mem` is never true, so the persist MR path is compiled but not taken.
  Confirms **no regression**; does NOT yet show the perf signal.

---

## Next steps

1. **GPU cluster run (the whole point).** Build the bench with CUDA and run on 2 GPU nodes:
   ```
   mpicc -O2 -DUSE_CUDA test/persist_mr_bench.c -o persist_mr_bench -lcudart
   mpiexec -n 2 ./persist_mr_bench <msg_bytes> <iters>
   ```
   - Use `msg_bytes` above `MPIR_CVAR_CH4_OFI_GPU_RDMA_THRESHOLD` so `need_mr` fires.
   - Compare `MPIR_CVAR_CH4_OFI_PERSISTENT_MR=1` vs `=0` (and vs the non-persistent line).
   - Expectation: persistent run drops the per-start `fi_mr_reg`/`fi_close`
     (dmabuf export + GPU pin) — should pull ahead as message size / iteration count grow.
   - Sanity: confirm "CORRECT" prints in both CVAR modes.
2. **Decide** based on the signal whether to invest further (Regime B host rendezvous,
   anysource, upstreaming).
3. Cleanup if pursuing: remove the redundant duplicate `MPIDI_NM_prequest_free_hook`
   prototype in `ofi_impl.h` (single source of truth is now `mpidpre.h`).
4. Broader correctness: run the MPICH pt2pt persistent test suite with the CVAR on and off.

## Build recipe (reference)

```
./autogen.sh
./configure --prefix ~/opt/mpich --with-libfabric=~/opt/libfabric
make -j6
```
Device resolves to `ch4:ofi`. GPU support auto-enables when a CUDA/ZE/HIP runtime is
present (disabled on the plain dev host, which is why local runs don't exercise Regime A).

## Notes / loose ends

- No commits made; working-tree only.
- Redundant `MPIDI_NM_prequest_free_hook` prototype in both `mpidpre.h` and `ofi_impl.h`
  (harmless; dedupe when tidying).
- Separate, unrelated: MPICH has a `.gitignore` gap for in-source-build generated files
  (`Version`, `src/env/mpi*.{bash,sh}.in`, `mpich.pc`, `mpich.module`) — candidate to
  report upstream.
