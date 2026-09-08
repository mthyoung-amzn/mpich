# Design: Per-Persistent-Request MR Caching in the CH4 OFI Netmod

Status: **DRAFT for review**
Scope: CH4 device, OFI netmod. Point-to-point persistent requests
(`MPI_Send_init` / `MPI_Recv_init` family). Partitioned and persistent
collectives are out of scope for v1.

---

## 1. Motivation

MPI persistent requests bind a fixed argument list (`buf`, `count`, `datatype`,
peer, tag, comm) to a request once, then `MPI_Start` / `MPI_Wait` it repeatedly.
The MPI standard (MPI-4.1/5.0 §4.9) explicitly frames this as an optimization
surface:

> "a communication with the same argument list (with the exception of the buffer
> contents) is repeatedly executed ... it may be possible to optimize the
> communication by binding the list of communication arguments to a persistent
> communication request."

and, in the advice to users:

> "Persistent request handles may bind internal resources such as MPI buffers ...
> for providing efficient communication. Therefore, it is highly recommended to
> explicitly free inactive request handles."

For EFA/libfabric, the most valuable reusable resource is a **memory region
(MR)**. Registering an MR (`fi_mr_reg` / `fi_mr_regattr` → `ibv_reg_mr` or
`ibv_reg_dmabuf_mr`) is expensive, and the buffer address/length that an MR
covers is *invariant across starts* of a persistent request (only the buffer
*contents* change). That invariance is the entire basis of this design: the
persistent request **is** the cache key, so no lookup/eviction machinery is
needed.

### Current behavior (verified)

Persistent pt2pt requests today store only their arguments
(`MPIDI_prequest_t`, `mpidpre.h`). On each `MPI_Start`,
`MPIDI_prequest_start` (`ch4_startall.h`) spawns a **fresh ephemeral**
`real_request` via `MPID_Isend` / `MPID_Irecv` and follows the normal path.
No resource is reused across starts.

There are two distinct MR-registration regimes in the OFI netmod:

- **Regime A — GPU/HMEM direct send/recv.** `MPIDI_OFI_send_normal`
  (`ofi_send.h`) and `MPIDI_OFI_do_irecv` (`ofi_recv.h`) call
  `MPIDI_OFI_register_memory_and_bind`, which goes through the **global MR
  cache** (`mr_cache.c`). This only triggers for device buffers with
  `MR_HMEM` above `GPU_RDMA_THRESHOLD`.
- **Regime B — rendezvous RDMA (host buffers).** For `data_sz > EAGER_THRESH`,
  `ofi_rndv_read.c` / `ofi_rndv_write.c` register per-NIC MRs in
  `prepare_rdma_info` (`ofi_rndv_rdma_common.inc`, comment: *"fi_mr_reg is
  expensive, distribute over num_nics"*) and then **unconditionally
  `fi_close`** every MR after each transfer (`rndvread_ack_event` /
  `rndvwrite_ack_event`). This path does **not** use the cache at all.

### The global MR cache is rudimentary

`mr_cache.c`: static array (`mr_cache[1024]`, effective cap
`MPIR_CVAR_CH4_OFI_MR_CACHE_SIZE = 16`), O(n) linear search, O(n) LRU eviction,
`memmove` array-shift on delete, keyed on exact `base_addr + ctx_idx +
buffer_id`. Stale-overlap detection is `#ifdef`'d out. It is aimed at HMEM/GPU.

### Host vs device registration cost

Both regimes ultimately call rdma-core, but the device path does strictly more
work per registration (`prov/efa/src/efa_mr.c`, `efa_mr_reg_ibv_mr`):

- **Host buffer** → `ibv_reg_mr(pd, base, len, access)` — one verbs call.
- **Device buffer** → dmabuf path: `ofi_hmem_get_dmabuf_fd` (GPU-driver call) +
  `ibv_reg_dmabuf_mr` + `ofi_hmem_put_dmabuf_fd`, with fallback to `ibv_reg_mr`.

So caching helps **both** regimes, for different reasons:

- **Regime B (host rendezvous):** there is *no* caching today — register + close
  on *every* start, per NIC. Even at host prices this is pure waste in a tight
  loop.
- **Regime A (device GDR):** per-op registration is genuinely expensive *and*
  the existing global cache is small and evictable (default cap 16), so a
  long-lived persistent request can have its MR evicted out from under it.

**Goal:** attach the MR to the persistent request so it is registered once (on
first start) and reused on every subsequent start, deregistered at
`MPI_Request_free`, bypassing the global cache entirely for persistent requests.

---

## 2. Design principles

1. **The persistent request is the cache key.** No hashing, no eviction, no LRU.
   One MR (or per-NIC MR set) lives with the request from first `MPI_Start`
   until `MPI_Request_free`. Reuse is an O(1) validity check.
2. **Keep OFI out of shared code.** Shared CH4 code sees only an *opaque,
   netmod-owned* handle. No OFI types cross into `ch4_*.h`.
3. **Performance first.** No virtual dispatch. The hot path per start is one
   NULL-check plus one pointer-equality validity check. Non-persistent traffic
   and eager sends pay nothing.
4. **Forward-thread the handle (option "a").** The handle is populated in
   `prequest_start` and passed *into* the operation, so it is present before any
   protocol (eager / rendezvous / GDR-direct) issues anything. This avoids a
   race where a synchronous GDR-direct registration could fire before a
   back-pointer could be installed.
5. **Extensible to other netmods.** The handle is a tagged base struct; other
   netmods can define their own derived state or ignore it.

---

## 3. Data structures

### 3.1 Shared base (NM-agnostic)

Reuse the existing owner discriminator from `ch4_types.h`:

```c
/* existing, ch4_types.h */
enum { MPIDI_NETMOD = 0, MPIDI_SHM = 1 };
```

Only one netmod is compiled into a given binary (`BUILD_CH4_NETMOD_OFI` xor
`_UCX`), so the only runtime heterogeneity is netmod-vs-SHM — which this enum
already expresses. No new enum is introduced.

```c
/* shared ch4 (mpidpre.h) — forward-declared; NM-agnostic */
typedef struct MPIDI_NM_persist_base {
    int owner;    /* MPIDI_NETMOD or MPIDI_SHM (from ch4_types.h) */
    /* room for future NM-agnostic fields (e.g. bound base/len for validity) */
} MPIDI_NM_persist_base_t;
```

### 3.2 OFI-private derived struct (base MUST be first member)

```c
/* ofi_pre.h — OFI-private */
typedef enum {
    MPIDI_OFI_PERSIST_NONE = 0,
    MPIDI_OFI_PERSIST_HMEM,   /* Regime A: single MR, bound on one ctx_idx */
    MPIDI_OFI_PERSIST_RNDV,   /* Regime B: per-NIC MR array + rkeys */
} MPIDI_OFI_persist_kind;

typedef struct {
    MPIDI_NM_persist_base_t base;   /* base.owner == MPIDI_NETMOD; MUST be first */
    MPIDI_OFI_persist_kind  kind;

    /* validity guard: re-register if these change */
    const void *reg_base;
    MPI_Aint    reg_len;

    int             num_nics;       /* HMEM: 1; RNDV: MPIDI_OFI_global.num_nics */
    struct fid_mr **mrs;            /* [num_nics] */

    /* RNDV only */
    uint64_t   *rkeys;              /* [num_nics], LOCAL_MR_KEY unless PROV_KEY */
    uint64_t    remote_base;        /* when MR_VIRT_ADDRESS */

    /* HMEM only */
    int         ctx_idx;            /* the (vci,nic) the single MR was bound on */
} MPIDI_OFI_persist_mr_t;
```

Because `base` is the first member, `(MPIDI_NM_persist_base_t *)ofi_ptr` and
`(MPIDI_OFI_persist_mr_t *)base_ptr` are both valid (C guarantees no leading
padding; a pointer to a struct equals a pointer to its first member). Downcasts
are **guarded** by `base->owner`, never blind.

### 3.3 Shared prequest struct gains one opaque pointer

```c
/* mpidpre.h, MPIDI_prequest_t */
typedef struct MPIDI_prequest {
    MPIDI_ptype p_type;
    void *buffer;
    MPI_Aint count;
    int rank, tag, context_offset;
    MPI_Datatype datatype;
    MPIDI_NM_persist_base_t *nm_persist;   /* NEW: opaque; NULL until first start */
} MPIDI_prequest_t;
```

The handle is heap-allocated (**not** in the `dev.ch4` union, which is overlaid
with the ephemeral request's netmod area).

---

## 4. Carrier seam (how the handle reaches the netmod)

The netmod dispatch boundary is `MPIDI_NM_mpi_isend` / `MPIDI_NM_mpi_irecv`.
This design adds **one** opaque parameter there. This mirrors the existing
precedent on the recv side, which already carries a trailing
`MPIR_Request *partner` (the `MPI_ANY_SOURCE` sibling request) that shared code
forwards to the netmod without interpreting.

`partner` and `nm_persist` are kept as **two distinct parameters**:

- `partner` — **per-operation** anysource sibling; read on the recv fast path of
  every recv (usually NULL); wired via `MPIDI_REQUEST_SET_LOCAL`.
- `nm_persist` — **per-request** cached MR state; consulted only at registration
  sites.

They are *not* merged. They have different lifetimes (per-op vs per-request),
different hotness (always-read vs rarely-read), and — importantly — they are
**not logically mutually exclusive**. They only appear mutually exclusive in v1
because we punt on anysource (see §7). A future anysource-persistent recv would
have *both* live at once (a matched sibling *and* a cached recv-buffer MR), so we
deliberately do not add an assertion coupling them, and we do not union them.

### Signatures touched

Shared CH4 (forwards an opaque pointer; non-persistent callers pass `NULL`):

- `MPIDI_isend` (`ch4_send.h`), `MPIDI_irecv` (`ch4_recv.h`)
- `MPID_Isend` / `MPID_Irecv` (`ch4_send.h` / `ch4_recv.h`) — the
  `prequest_start` caller passes `preq->nm_persist`; all others pass `NULL`.

Netmod dispatch (one opaque, netmod-defined pointer — the correct seam):

- OFI: `MPIDI_NM_mpi_isend` (`ofi_send.h`), `MPIDI_NM_mpi_irecv` (`ofi_recv.h`)
- UCX: `ucx_send.h`, `ucx_recv.h` — signature only; ignored in v1
- AM fallback: `netmod_am_fallback_send.h`, `netmod_am_fallback_recv.h` —
  signature only
- SHM: `MPIDI_SHM_mpi_isend` / `irecv` — signature only; SHM has no NIC MRs

OFI-internal (where the handle is consumed):

- `MPIDI_OFI_send` and the rndv / GDR-direct registration sites.

---

## 5. Lifecycle

### 5.1 Allocation (lazy, netmod-owned)

On the **first** `MPI_Start` that would register, `prequest_start` obtains a
handle via an NM hook:

```c
/* ch4_startall.h, MPIDI_prequest_start, before dispatch */
if (MPIDI_PREQUEST(preq, rank) != MPI_ANY_SOURCE &&   /* anysource opted out, see §7 */
    MPIDI_PREQUEST(preq, nm_persist) == NULL) {
    MPIDI_PREQUEST(preq, nm_persist) = MPIDI_NM_persist_alloc();  /* NULL if NM opts out */
}
...
MPID_Isend(buf, ..., attr, &preq->u.persist.real_request, MPIDI_PREQUEST(preq, nm_persist));
```

`MPIDI_NM_persist_alloc` (OFI) allocates and zeroes an
`MPIDI_OFI_persist_mr_t`, sets `base.owner = MPIDI_NETMOD`,
`kind = MPIDI_OFI_PERSIST_NONE`. UCX/SHM return `NULL` (no-op participation).

Allocation is lazy so persistent requests that never hit a registering-size
operation pay nothing.

### 5.2 Consume (both regimes)

At each registration site, guard on the handle and its owner:

```c
MPIDI_OFI_persist_mr_t *slot = NULL;
if (nm_persist && nm_persist->owner == MPIDI_NETMOD)
    slot = (MPIDI_OFI_persist_mr_t *) nm_persist;
```

**Regime A — `MPIDI_OFI_send_normal` / `MPIDI_OFI_do_irecv`:**

```c
if (slot) {
    mr = MPIDI_OFI_persist_get_or_reg(slot, MPIDI_OFI_PERSIST_HMEM,
                                      base, data_sz, &attr, ctx_idx);
    /* reuse if slot->kind==HMEM && slot->reg_base==base && slot->ctx_idx==ctx_idx;
     * else register once (register_memory + mr_bind), store in slot.
     * Does NOT insert into the global mr_cache. */
} else {
    MPIDI_OFI_register_memory_and_bind(base, data_sz, &attr, ctx_idx, &mr);  /* unchanged */
}
```

The completion path (`ofi_events.h`) must **not** free a prequest-owned MR:
if the ephemeral request carries a persist slot, skip the
`MPIDI_OFI_mr_complete` call entirely (ownership belongs to the persistent
request, released at free). Since the MR was never inserted into the global
cache, `mr_complete` would otherwise `fi_close` it.

**Regime B — `prepare_rdma_info`:**

```c
if (slot && persist_valid(slot, MPIDI_OFI_PERSIST_RNDV, data, data_sz)) {
    /* reuse: copy slot->mrs[]/rkeys[]/remote_base into hdr; skip fi_mr_reg loop */
} else {
    /* register per-NIC as today; if slot, move mrs[]/rkeys[] into slot
     * (instead of a transient array) and record reg_base/reg_len/num_nics. */
}
```

The `rndv{read,write}_ack_event` teardown becomes: **if MRs are
prequest-owned, do not `fi_close` and do not free rkeys** — only release the
transient control request / hdr. Non-persistent rendezvous is unchanged.

### 5.3 Free (one netmod hook)

```c
/* ch4_request.h, MPID_Prequest_free_hook */
MPL_STATIC_INLINE_PREFIX void MPID_Prequest_free_hook(MPIR_Request * req) {
    MPIDI_NM_prequest_free_hook(req);                              /* NEW: no-op for ucx/shm */
    MPIR_Datatype_release_if_not_builtin(MPIDI_PREQUEST(req, datatype));
}
```

`MPIDI_NM_prequest_free_hook` (OFI): if `nm_persist` is set and owned by
`MPIDI_NETMOD`, `fi_close` each `mrs[i]`, return each rkey via
`MPIDI_OFI_mr_key_free` (unless `PROV_KEY`), free `mrs`/`rkeys` and the struct,
NULL the slot.

The MPI standard guarantees a request is freed only when **inactive**, so no MR
is in flight at free time. Cleanup runs under the VCI lock already held during
request teardown.

---

## 6. Correctness guards

- **Validity check** (`reg_base == buf && reg_len == data_sz`): defends against
  the legal-but-unusual case where the effective registered range differs from
  what the slot cached. On mismatch: tear down the stale slot MR(s) and
  re-register.
- **Protocol / side volatility.** The rendezvous protocol (read vs write vs
  direct vs pipeline) is chosen per start from the CTS and peer packing needs,
  and read-vs-write decides *which side registers* (and the access flags:
  `FI_REMOTE_READ` vs `FI_REMOTE_WRITE`). The slot records `kind` and access
  intent; if a later start needs a different kind/side/access than cached, treat
  it as invalidation → deregister + re-register. Steady-state loops never hit
  this; it is a safety fallback, not the hot path. **Never** reuse a
  `FI_REMOTE_READ` MR where `FI_REMOTE_WRITE` is required.
- **`num_nics` / VCI stability.** A persistent request binds a VCI at init and
  `num_nics` is fixed post-init, so the per-NIC array sizing is stable for the
  request's lifetime. (Assumption to confirm during review.)
- **Owner tag.** Every downcast checks `base->owner == MPIDI_NETMOD`. Even if
  anysource is enabled later (§7) and a handle could reach SHM, SHM sees an
  owner it does not recognize and declines to interpret it.

---

## 7. `MPI_ANY_SOURCE` — opted out in v1

Anysource receives are dispatched to **both** SHM and NM concurrently
(`anysource_irecv`, `ch4_recv.h`), with the first match cancelling the sibling
via the `anysrc_partner` link. Caching an MR on such a request is *logically*
fine — the receive buffer is bound at `_init` and invariant across starts,
independent of which source matches — but it raises a "which transport
completed this start" question that we choose not to solve in v1.

**v1 behavior:** if `rank == MPI_ANY_SOURCE`, leave `nm_persist == NULL`
(allocation skipped in `prequest_start`). Such requests transparently fall back
to today's per-op registration path.

This exclusion is a **product decision, not a logical invariant.** A future
anysource-capable version would legitimately have both a live `partner` and a
live `nm_persist` at once; nothing in this design forecloses that (hence: no
assertion coupling them, no merging of the two parameters, and an owner-tagged
base struct with room for per-transport sub-state).

---

## 8. CVAR

```
MPIR_CVAR_CH4_OFI_PERSISTENT_MR   (boolean, default: on)
```

When off, persistent requests use the existing behavior (global cache for
Regime A; per-transfer register/close for Regime B). Allows A/B measurement and
a kill switch if a provider misbehaves.

---

## 9. Implementation plan

1. **Scaffolding (shared + hooks).**
   - `MPIDI_NM_persist_base_t` in `mpidpre.h`; `nm_persist` field in
     `MPIDI_prequest_t`.
   - NM interface: `MPIDI_NM_persist_alloc`, `MPIDI_NM_prequest_free_hook`
     (OFI real; UCX/SHM no-op).
   - Thread the `nm_persist` parameter through the isend/irecv dispatch
     signatures (shared + all netmods + fallback).
   - `MPID_Prequest_free_hook` calls the NM hook.
   - `prequest_start` lazy-allocs the handle (skip for anysource) and forwards
     it.
2. **Regime B (rendezvous) — the clean win.**
   - `MPIDI_OFI_persist_mr_t` + get-or-reg + validity.
   - Teach `prepare_rdma_info` to reuse; teach `rndv*_ack_event` to skip
     teardown for owned MRs.
3. **Regime A (GDR direct).**
   - Teach `send_normal` / `do_irecv` to register into the slot and bypass the
     global cache.
   - Teach `ofi_events.h` completion to skip `mr_complete` for owned MRs.
4. **CVAR + gating.**
5. **Tests / benchmarks** (see §10).

---

## 10. Testing

- **Correctness:** existing pt2pt persistent tests (`test/mpi/pt2pt`) must pass
  with the CVAR on and off. Add a test that starts/completes a persistent
  large-message send+recv many times and validates data each iteration
  (contents change, mapping reused).
- **MR lifetime:** verify (via provider counters or `track_mr`) that a
  persistent large-message loop performs **one** registration per NIC total,
  not one per start; and that MRs are released at `MPI_Request_free`, not
  before.
- **Regime coverage:** exercise both host rendezvous (Regime B) and, on a GPU
  node, device GDR (Regime A).
- **Anysource:** confirm a persistent anysource recv still works (falls back,
  `nm_persist == NULL`).
- **Perf:** microbenchmark persistent vs non-persistent large-message ping-pong;
  expect the per-start `fi_mr_reg`/`fi_close` pair (per NIC) to disappear.

---

## 11. Open items for review

1. Confirm the `num_nics` / VCI invariant (§6) — no runtime NIC-count change
   that would invalidate a cached per-NIC MR array.
2. Confirm the carrier seam choice: one added `MPIDI_NM_persist_base_t *`
   parameter on the NM isend/irecv dispatch (vs. bundling into a per-op context
   struct). Recommendation: the single parameter, mirroring `partner`.
3. Confirm Regime A completion-path change (skip `mr_complete` for owned MRs) is
   the cleanest interception point vs. an alternative in `mr_complete` itself.
4. Confirm v1 scope excludes partitioned (`Psend_init`) and persistent
   collectives.
