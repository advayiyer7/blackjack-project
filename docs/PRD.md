# PRD — blackjack-counter

> Generated 2026-07-02 via `/prp-prd` (BUILD-GUIDE Phase 2). Inputs: BUILD-GUIDE §1 objective,
> `docs/research/trainer-notes.md`, `docs/research/stack-notes.md`, `docs/research/strategy-tables.md`.
> Success metrics below are the BUILD-GUIDE Phase 8 gates verbatim — they are the acceptance
> criteria for the whole project.

## Problem Statement

Advay is learning Hi-Lo card counting and needs deliberate practice with immediate, trustworthy
feedback: while playing a free blackjack trainer, he wants to know the running count, true count,
and count-adjusted optimal action at any moment, without taking his eyes off the table. Separately
(and equally important), he needs a portfolio project that demonstrates an end-to-end applied-AI
pipeline — data collection → model fine-tuning → real-time inference → stateful tracking →
decision engine → RL — with honest, measurable evals. One project solves both.

## Evidence

- The Wizard of Odds v2 trainer's own "warn on mistakes" feature only flags basic-strategy errors —
  it does not teach count deviations (Illustrious 18 / Fab 4), which are exactly the hard part of
  counting practice (confirmed from trainer source, `docs/research/trainer-notes.md` §8).
- The trainer displays its own running/true count (`#countinfo`), which proves count-practice
  demand and — critically — gives this project a built-in ground-truth oracle for evals
  (trainer-notes §5).
- Portfolio-value assumption: hiring signals favor projects with real CV pipelines, quantified
  accuracy gates, and honest failure reporting over toy demos. Assumption — validated only by the
  finished repo.

## Proposed Solution

A local Windows app: on a global hotkey it screenshots the user-selected trainer region, detects
all visible cards with a YOLO model fine-tuned on the trainer's fixed sprite art, reconciles table
state across frames (new cards, round boundaries, splits, hole-card reveal), maintains Hi-Lo
running/true count, and shows the count-adjusted optimal action in an always-on-top overlay.
A vision pipeline is chosen deliberately over DOM scraping (which the trainer would permit) because
the CV pipeline is the portfolio point; DOM/`#countinfo` scraping is reserved for eval ground truth.
An RL chapter trains a tabular agent on a count-aware blackjack environment and benchmarks it
against the deterministic strategy tables.

## Key Hypothesis

We believe a hotkey-driven CV advisor with a provably-correct logic core will make count-deviation
practice fast and trustworthy for a solo learner. We'll know we're right when the Phase 8 gates
pass: ≤1 of 20 recorded shoes shows any count drift, detection accuracy ≥99.5%, and 100% of
displayed actions match the verified strategy tables given the tracked state.

## What We're NOT Building

- **Anything for real-money play** — violates casino ToS everywhere and device statutes in some
  jurisdictions; also mathematically pointless online (per-hand shuffle RNG, shallow live-dealer
  penetration). README states this plainly.
- **Multi-site support** — v1 targets the WoO v2 trainer's fixed sprite art only. The detector
  interface is swappable, but no second site ships in v1.
- **Automated bet sizing / bankroll management UI** — only a TC-indexed bet suggestion in the
  overlay, nothing more.
- **Auto-play / botting** — the app advises the human; it never clicks, types into, or manipulates
  the game.
- **Automatic shuffle detection** — v1 uses a manual reset hotkey (plus the "LAST HAND" badge as a
  cue); auto-detect is a documented stretch goal.
- **Suit-level game logic** — suits are kept as detection classes for robustness, but all game
  logic is rank-only.

## Success Metrics (= BUILD-GUIDE Phase 8 gates + milestone gates)

| Metric | Target | How Measured |
|--------|--------|--------------|
| End-to-end count integrity (pass^k) | Zero count drift in ≥19/20 full shoes; any drifting shoe gets a root-cause note | 20 recorded shoes at the trainer, predicted RC vs `#countinfo` ground truth each round → `docs/EVAL.md` |
| Card-detection accuracy (e2e) | ≥99.5% | Same 20-shoe eval, per-card comparison |
| Decision correctness | 100% match vs M1 strategy table given tracked state | Same 20-shoe eval |
| Detector quality (held-out) | mAP50 ≥0.99, per-rank recall ≥0.995 | `models/metrics.json` from held-out split at M4 |
| Inference latency | <100 ms/frame CPU (ONNX) | Local benchmark script at M4 |
| Overlay latency | <300 ms hotkey→display | M6 live smoke test, 2 full shoes, no crashes |
| Strategy engine parity | 100% of ~340 chart cells + deviation overrides | M1 tests generated from `strategy-tables.md` |
| Counting engine soundness | Full-shoe RC sums to 0; 10k simulated shoes zero drift | M2 property tests |
| Test coverage | ≥80% overall; ≥95% strategy/counting; ≥85% tracker | `pytest --cov` |
| RL agreement | ≥97% action agreement with basic strategy in TC=0 bucket; each detected deviation honestly compared to its published index | `rl/compare.py` → `docs/RL-REPORT.md` |

## Open Questions

- [ ] Trainer's split-count quirk: `player.js` adds `CountValue[10]` to the running count on every
      split — must live-verify at M3 whether the displayed count actually shifts on splits, else
      20-shoe evals will show false drift (trainer-notes §9.7).
- [ ] Three deviation indices remain UNRESOLVED in research (Fab 4 15vA H17 index; I18 H17 shifts
      for 10vA, 12v6) — engine ships the confirmed S17-published values with the documented H17
      carve-outs; RL chapter may produce evidence on these.
- [ ] How much the page chrome/ads shift the table position between sessions (affects M3 capture
      reproducibility; M6 region selector absorbs it at runtime).
- [ ] H17 vs S17 for the reference eval config: trainer defaults to H17 — v1 evals run H17; the
      engine supports both via config.

---

## Users & Context

**Primary User**
- **Who**: Advay — CS student, comfortable with Python and the command line, learning Hi-Lo
  counting; also the project's author presenting it to recruiters.
- **Current behavior**: plays the WoO trainer with its count display on, mentally counting;
  checks deviation charts by hand between rounds.
- **Trigger**: mid-shoe, wants instant confirmation of the count and the correct (possibly
  deviation) play without breaking flow.
- **Success state**: presses one hotkey, sees RC / TC / decks remaining / recommended action
  (flagged if it's a deviation) / TC-indexed bet suggestion in <300 ms, and trusts it.

**Job to Be Done**
When I'm practicing counting at the trainer, I want on-demand verified count-and-action feedback,
so I can build accurate counting habits faster than manual chart-checking allows.

**Non-Users**
Real-money players (explicitly excluded, see Not Building), multi-site users, anyone wanting an
auto-player.

---

## Solution Detail

### Core Capabilities (MoSCoW)

| Priority | Capability | Rationale |
|----------|------------|-----------|
| Must | Strategy engine: full 6D basic strategy (H17+S17) + I18 + Fab 4, pure functions | The correctness core; every displayed action derives from it |
| Must | Counting engine: Hi-Lo RC, floored TC, decks-remaining, shoe simulator | Provable before any pixel is read |
| Must | Dataset pipeline: region capture + sprite template auto-labeler | Feeds the detector; sprite sheet gives free ground-truth templates |
| Must | YOLO detector fine-tuned on trainer art, ONNX CPU inference | The portfolio CV centerpiece |
| Must | State tracker: frame diff → new cards, round boundaries, splits, hole-card reveal, manual shuffle reset | The genuinely hard part; a silently wrong count is the worst failure mode |
| Must | App shell: region selector, hotkeys (capture/reset/quit), always-on-top overlay | The user-facing loop |
| Must | RL chapter: count-aware env, tabular agent, divergence report | The differentiating portfolio story |
| Should | Low-confidence surfacing in overlay (never guess silently) | Trust requirement; silent-failure review gate |
| Should | TC-indexed bet suggestion | Cheap, rounds out the counting workflow |
| Could | Auto shuffle detection ("LAST HAND" badge or count discontinuity) | Stretch; manual reset ships first |
| Won't | DOM-scraping gameplay pipeline, multi-site, auto-play, real-money anything | See Not Building |

### MVP Scope

M1+M2 (pure logic core, fully tested) + M3–M6 (vision → tracker → app shell) against the WoO v2
trainer at 6 decks / 75–80% penetration / H17 defaults. RL chapter completes the portfolio story.

### User Flow

1. Start app → drag-select the trainer's game region once (persisted).
2. Play the trainer; press the capture hotkey **at least once at each round's settle** (that
   single capture keeps the count exact — see ARCHITECTURE §4.5), plus at any decision point
   where advice is wanted. The overlay warns if a round appears to have been skipped unsettled.
3. Overlay updates: RC, TC, decks left, action (deviation-flagged), bet suggestion.
4. "LAST HAND" badge appears → finish hand → press reset hotkey at the new shoe.
5. Quit hotkey exits cleanly. Zero network calls the whole time.

---

## Technical Approach

**Feasibility**: HIGH — de-risked by Phase 1: card art is a fixed 67×94 sprite sheet on a fixed
960×640 canvas (near-ideal detection conditions); the count oracle exists; APIs for every stack
piece verified with working snippets (`stack-notes.md`).

**Architecture Notes** (locked in Phase 3, these are the PRD-level commitments)
- Pure logic core (`strategy/`, `counting/`) with zero I/O — provable via exhaustive/property tests.
- Frozen/immutable dataclasses; new state per frame; side effects only at the edges (`app/`).
- Detector behind a swappable protocol — template-matcher fallback implements the same interface,
  unblocking M5 before M4's weights exist.
- Runtime makes zero network calls; only Colab notebook has egress (labeled crops up, weights down).
- `pynput` registers named hotkeys only — structurally incapable of keylogging (security rule §4.1).

**Technical Risks**

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Trainer split-count quirk causes false eval drift | M | Live-verify at M3 (open question #1); if real, mirror it in the eval harness only — never in our count |
| Silent reshuffle breaks tracker state | H (by design — trainer has no shuffle animation) | Manual reset hotkey + "LAST HAND" badge protocol documented in overlay; auto-detect stretch |
| Windows DPI scaling misaligns capture region vs overlay | M | `SetProcessDpiAwareness(2)` first thing in main(), before mss and tkinter (stack-notes §3) |
| Page chrome/ads shift table position between sessions | M | Region selector re-run per session; selector-based crop, never fixed screen coords |
| Detector underperforms on overlapped fanned cards (24px offsets) | L–M | Dataset must include heavy fan/split coverage; gate is per-rank recall ≥0.995 — if missed, fix dataset, not model |
| Colab/local ONNX opset mismatch | L | Pin `opset=17` at export (stack-notes §1) |

---

## Implementation Phases

Mirrors BUILD-GUIDE Phases 3–9 (Phases 0–2 complete). Statuses live here; the BUILD-GUIDE remains
the authoritative process doc.

| # | Phase | Description | Status | Parallel | Depends | PRP Plan |
|---|-------|-------------|--------|----------|---------|----------|
| 1 | Architecture (BG Phase 3) | Dataclass contracts, reconciliation algorithm, detector protocol → docs/ARCHITECTURE.md | pending | - | - | - |
| 2 | M1 Strategy engine | Tables + deviations + engine, tests generated from strategy-tables.md | pending | - | 1 | - |
| 3 | M2 Counting engine | Hi-Lo, TC, shoe simulator, property tests | pending | with 4 | 1 | - |
| 4 | M3 Dataset pipeline | Capture + auto-label ~1,500–2,500 frames, all 52 classes ≥40 instances | pending | with 3 | 1 | - |
| 5 | M4 Detector fine-tune | Colab T4 train, ONNX export, held-out eval | pending | with 6 (Colab runs in background) | 4 | - |
| 6 | M5 State tracker | Reconciliation vs recorded shoes, exact count replay | pending | with 5 (uses template-matcher fallback) | 2, 3 | - |
| 7 | M6 App shell | Region selector, hotkeys, overlay; live smoke test | pending | - | 5, 6 | - |
| 8 | RL chapter (BG Phase 5) | Env + tabular agent + divergence report | pending | - | 2, 3 | - |
| 9 | Review/Security/Verify/Ship (BG Phases 6–9) | Full-repo reviews, security scan, 20-shoe eval, README/ship | pending | - | 7, 8 | - |

### Parallelism Notes

M2 (pure logic) and M3 scaffolding touch disjoint trees (`counting/` vs `vision/`) — safe to
overlap. M4's Colab training is a background wait — M5 proceeds against the template-matcher
fallback (same protocol) meanwhile. The RL chapter depends only on the logic core, so it can start
any time after M1+M2 if the vision track stalls.

---

## Decisions Log

| Decision | Choice | Alternatives | Rationale |
|----------|--------|--------------|-----------|
| Detection approach | Fine-tuned YOLO on fixed sprites | DOM scraping (easier, trainer permits it) | CV pipeline is the portfolio point; DOM reserved for eval oracle |
| Reference rule config | Trainer defaults: 6D, H17, DAS, LS-any-card, 3:2 | S17 config | Trainer default; engine supports both; evals note exact config |
| TC convention | Flooring (round toward −∞) | Truncation, rounding | Schlesinger/CVData convention (strategy-tables §3) |
| Negative I18 boundary | Hit strictly below index; stand at index | "at or below" | Canonical convention; corroborated by "hit 12v4 when negative" (discrepancy #7) |
| Capture viewport | Force ≥1200 px browser width (0.9 scale) | 0.74 scale | Crisper cards, one known scale factor |
| Unresolved H17 indices | Ship confirmed values + documented carve-outs | Guess numbers | Honesty over invented precision; RL may add evidence |
| Shuffle handling v1 | Manual reset hotkey | Auto-detect | Trainer reshuffles silently; auto-detect is stretch |

---

## Research Summary

**Market/context**: The trainer itself only corrects basic strategy, not deviations; dedicated
counting-drill apps exist but none advise against a live third-party table view — the CV angle is
the novel (and portfolio-relevant) part. No further market research warranted for a portfolio
project (explicit non-goal: commercialization).

**Technical**: See `docs/research/` — trainer internals (source-level), stack API notes with
verified snippets, and cross-checked strategy tables ready to become test fixtures.

---

*Generated: 2026-07-02*
*Status: DRAFT — becomes final when Phase 3 architecture is user-confirmed*
