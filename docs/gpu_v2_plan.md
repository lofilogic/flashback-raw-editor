# GPU Pipeline V2 — Implementation Plan

> Status: **proposed**, not started. Written June 2026 from an analysis session
> (review of the shipped GPU-resident pipeline + a RapidRaw sanity-check).

## How to use this plan

These conclusions are **current best understanding, not law.** If, while
implementing, you see a better way — or evidence that contradicts a step here —
**say so and flag it.** Prefer "here's a possible improvement / here's what the
measurement shows" over silent adherence. Re-raising something noted as "ruled
out" is welcome; it costs little and the earlier call may have been wrong.

Confidence is marked per item. Treat **high** as "act on it," **medium** as
"likely right, verify against measurement before committing."

## Product philosophy this work must respect

Flashback is a **lean, focused, film-look** processor for the Flashback V2 camera
(+ secondary generic raw). **Aesthetics over pristine fidelity; fun over
state-of-the-art; pragmatic, no bloat.** Parity bar is **perceptual, not
bit-exact.** Do a refactor because it's worth it for the user (faster, simpler,
fewer bugs) — not because it's convention. If a step here starts to feel like
overengineering, that's a signal to stop and reconsider, not to push through.

## The diagnosis (high confidence)

The shipped pipeline keeps pixels GPU-resident but **still allocates per frame**:
every stage calls `_create_tex` (~96 MB per rgba32float texture at ~3000×2000; a
dozen+ per render → >1 GB transient allocation per frame during a slider scrub),
plus a fresh uniform buffer and bind group per stage. That allocation churn —
CPU-side driver work, not GPU compute — is the top remaining interactive cost.

External sanity-check (RapidRaw, Rust+wgpu): renders to a GPU surface with **no
readback for the live preview** (readback is export-only); creates pipelines/
textures **once** and reuses them; drives a single `AllAdjustments` struct via
`write_buffer`; **requires a GPU (no CPU fallback).** Independent confirmation of
the direction below.

---

## Phases (each independently shippable + parity-verifiable)

### Phase 1 — Per-render arena allocator  ·  confidence: high  ·  risk: low
**Goal:** eliminate steady-state per-frame allocation. Allocate textures/uniform
buffers from a bump arena that **resets at the end of each render**.

- **Arena, NOT global ping-pong** — reason: it preserves the write-once `Frame`
  invariant (`core/gpu.py` `Frame`, ~line 1257) that makes the color chain easy
  to verify. No Frame outlives its render, so the lazy `cpu()`/`gpu()` cache stays
  sound. (RapidRaw uses ping-pong + explicit named buffers instead — valid, but it
  has no `Frame` abstraction. Don't import that trade unless you also drop `Frame`.)
- Persistent uniform buffers written with `write_buffer`, not recreated per stage.
- Scope: `core/gpu.py` only. No behavior change intended.
- **Acceptance:** existing parity tests pass **AND** a new multi-frame
  dirty-arena test passes (see Traps #3) **AND** `tools/bench_pipeline.py` shows a
  scrub-path improvement on the slow (Windows/RTX3090) machine.

### Phase 2 — Table-drive `_build_pipelines`  ·  confidence: high  ·  risk: low
**Goal:** collapse ~300 lines of near-identical bind-group-layout boilerplate
(`core/gpu.py` ~113–409) into a small table; there are ~4 distinct layout shapes.

- Pure hygiene, no behavior change. Reduces the silent-miswiring defect surface.
- **Acceptance:** parity tests pass; line count down materially; layouts unchanged.

### Phase 3 — Fuse point-ops into one parameterized pass  ·  confidence: medium  ·  risk: medium
**Goal:** one main compute pass driven by a params struct for all **per-pixel**
ops; keep a separate pass **only** for neighborhood ops. (RapidRaw validates this
shape exactly.)

- **Fuse:** gain (wb·tint·ev), vignette, ACEScct encode, LUT, grain.
- **Keep separate:** separable blurs (softness/bloom/sharpen/edge), CNR bilateral,
  halation.
- **Bounded by color-space walls** (Trap #1) — do not fuse across the
  ACEScct-encode boundary.
- **Acceptance:** perceptual parity within the agreed tolerance (Trap #2),
  side-by-side visual confirmation, parity + dirty-arena tests pass.

### Phase 4 — Render-to-surface  ·  confidence: medium  ·  risk: high  ·  SEPARATE DECISION
**Goal:** present the preview to a wgpu surface to drop the **final** readback —
the biggest end-state win, but it requires the preview widget to become a GPU
surface (touches Qt) and is arguably a different project.

- Do a **feasibility spike first**; do not bundle with 1–3.
- **Open question:** this phase may favour **rgba16float** (filterable → free
  hardware bilinear) over the shipped **rgba32float**. The original direction memo
  proposed f16; f32 shipped for losslessness. Re-evaluate here — flag, don't assume.
- Likely only worth it if Phases 1–3 don't already make interaction feel seamless.

> Possible later, not scheduled: demote the numpy path from a live fallback to a
> **test-only oracle** (RapidRaw ships GPU-only). Only if the no-GPU case stops
> mattering for real users — flag when relevant, don't pre-commit.

---

## Traps (what a fresh session would otherwise re-discover the hard way)

1. **Color-space boundaries cap fusion.** Pre-LUT is linear ACEScg, post-LUT is
   display sRGB, with ACEScct-encode + LUT as a hard wall between
   (`core/processor.py` ~804, `core/effects.py` header). Vignette must run linear;
   grain must run display. Can't fuse across the wall.
2. **Fusion can break a tight parity tolerance.** GPU `fma` ≠ CPU separate
   multiply-add in the last bit; tetrahedral interp + BLAS accumulation order
   differ too. Parity bar is **perceptual, not bit-exact** — set `atol`
   deliberately before fusing, don't chase float noise as if it were a regression.
3. **Parity tests don't cover buffer reuse.** They run single renders from clean
   state, so they can't see stale data leaking from a recycled arena/pool buffer
   (e.g. a stage that doesn't fully overwrite its dst, like the small bloom
   texture). Add a **multi-frame, dirty-arena parity test** before relying on
   Phase 1/3.
4. **Halation is load-time-cached on purpose.** It depends only on the raw, not on
   sliders (`core/processor.py` ~723, baked into the cached intermediate). Don't
   fold it into the live stage list — that's a correct asymmetry, not a cleanup.
5. **No tiling needed.** Binning lands images ~3000×2000, under the 8192 texture
   limit. Don't add RapidRaw-style tiling — binning designs that problem away.

## Verify with
`FLASHBACK_DEBUG_TIMING=1` + `tools/bench_pipeline.py <dng> --runs 5`, on both the
M3 and the slow Windows/RTX3090 box (the latter is the one that actually hurts).

## Related memory
`project-philosophy`, `gpu-v2-architecture-decisions`, `session-handover-gpu-resident`,
`perf-gpu-resident-direction`.
