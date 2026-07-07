---
name: blackjack-counter-sop
description: Standard operating procedure for the blackjack-counter project — pipeline order, architecture invariants, commands, data/training SOP, review discipline, trainer facts, and human-in-the-loop registry. Use at the START of any session working in this repo, before writing code, running the data pipeline, or resuming a milestone.
---

# blackjack-counter — Standard Operating Procedure

Real-time CV card-counting advisor for the Wizard of Odds free blackjack trainer
(portfolio project). Hotkey → screenshot region → detect cards (YOLO) → reconcile
table state → Hi-Lo running/true count → basic strategy + I18/Fab 4 deviations →
always-on-top overlay. **Ethics framing is non-negotiable**: training tool for
free-play simulators only; never for real-money play (README states this).

## 0. Read order (before any work)

1. `BUILD-GUIDE.md` — the runnable build prompt; phases, exit criteria, security profile.
2. `docs/PRD.md` — scope, success metrics (= Phase 8 gates), **milestone status table** (§Implementation Phases — keep it updated).
3. `docs/ARCHITECTURE.md` — every decision is binding. **§14 is the amendments table**: any deviation made during implementation MUST be recorded there. §13 is the original review record.
4. `docs/research/` — trainer-notes.md (geometry/mechanics source of truth), strategy-tables.md (M1 test fixture source), stack-notes.md (API pins).
5. Project memory (`blackjack-counter-status.md` in the Claude memory dir) — current position + pending human steps.

## 1. Pipeline position and status

`RESEARCH → PRD → ARCHITECTURE → TDD (M1–M6) → RL → REVIEW → SECURITY → VERIFY → SHIP`

Status lives in two places only: the PRD milestone table (authoritative) and the
project memory (session-to-session pointer). As of 2026-07-07: M1/M2 complete,
M3 train-pool complete (real capture ongoing), M5 core complete (recorded-shoe gate
pending), M4 tooling ready (awaiting user Colab run), M6/RL not started.

## 2. Architecture invariants (violating any of these = stop and re-read ARCHITECTURE)

- **Pure core, effectful edges.** `strategy/`, `counting/`, `tracker/` are pure,
  stdlib-only, frozen `@dataclass(frozen=True, slots=True)` everywhere, no I/O, no
  numpy. Side effects live only in `vision/` and `app/`.
- **Dependency direction:** `app → {vision, tracker, counting, strategy, types}`;
  `tracker → types` ONLY (no counting, no vision import — worker applies revealed
  ranks to ShoeState); `rl → {counting, strategy, types}`; everything imports
  `types.py`, which imports nothing.
- **Never guess silently.** Any untrustworthy frame → SUSPECT: `accepted=False`,
  nothing counted, warning surfaced. A skipped frame is recoverable; a silently
  wrong count is the project's worst failure mode. Every classifier change must be
  argued in the fail-safe direction and pinned by an adversarial test.
- **Position-free count arithmetic, position-aware boundaries** (§4). Count =
  multiset delta of (rank, suit) faces, but only after the transition classifier
  positively established DUPLICATE / NEW_DEAL / CONTINUATION / ROUND_ENDED.
- **Exact integer true count:** `(RC · 52) // remaining_cards`, half-deck clamp 26.
  Float decks value is overlay display ONLY (a float path caused a real off-by-one — §14 M2).
- **Geometry constants are duplicated by design** in `tracker/state.py` and
  `vision/synthesize.py` (tracker can't import vision). A non-skipped drift test in
  `tests/tracker/test_state.py` keeps them identical. Change both or neither;
  source of truth is trainer-notes §4/§6.
- **File discipline:** 200–400 lines typical, 800 hard max; functions <50 lines.

## 3. Per-milestone workflow (how work ships here)

1. Re-read the relevant ARCHITECTURE sections + research notes.
2. **TDD**: tests first (fixtures from research docs / synthetic builders), then
   implementation. Adversarial cases before happy paths for tracker/count code.
3. Run: full `pytest` + coverage (gates: strategy/counting ≥95%, tracker ≥85%,
   overall ≥80%) + `ruff check src tests scripts` (must be clean; do NOT reformat
   pre-existing files with `ruff format` — repo predates it).
4. **Review pass**: launch `code-reviewer` and `python-reviewer` agents in parallel
   on the new/changed files. Fix all CRITICAL/HIGH before commit (they have found
   real count-corruption bugs twice — do not skip this).
5. Record any contract/spec deviation as a new row in ARCHITECTURE §14; update the
   PRD milestone table status.
6. Conventional commits (`feat:`/`fix:`/`docs:`/`chore:`), **no attribution
   trailers** (user has attribution disabled; match existing history). Push to
   `https://github.com/advayiyer7/blackjack-project.git` (no `gh` CLI on this machine).
7. Update the project memory file with position + pending human steps.

## 4. Commands cookbook (run from repo root; venv: `.venv/Scripts/python.exe`)

| Task | Command |
|---|---|
| Full test suite | `.venv/Scripts/python.exe -m pytest -q` |
| With coverage gates | `.venv/Scripts/python.exe -m pytest -q --cov --cov-report=term` |
| Lint | `.venv/Scripts/python.exe -m ruff check src tests scripts` |
| Capture session (user plays) | `python scripts/capture_session.py` |
| Refresh dataset report | `python scripts/dataset_report.py` → `data/REPORT.md` |
| Regenerate synthetic pool | `python scripts/make_synthetic.py` (seed 7, per-class 80) |
| Build Colab dataset zip | `python scripts/export_dataset.py` → `data/dataset_yolo.zip` |
| Check M4 gates (needs weights) | `python scripts/eval_detector.py` (fallback: `--onnx models/best_960.onnx --imgsz 960`) |

Slow tests: `tests/tracker/test_replay.py` + real-asset round-trips need
`data/assets/{deck,table}.png` (gitignored; provenance + re-fetch URLs in
`data/assets/README.md`) and take ~1–2 min — they skip cleanly when assets are absent.

## 5. Dataset & training SOP (M3/M4)

**Pools:** synthetic frames (`data/synthetic/`, regenerable, gitignored) = TRAINING;
real captures (`data/raw/`, gitignored) = VAL/TEST. Never mix without recording the
decision in ARCHITECTURE §14.

**Capture protocol (user):** incognito browser (an extension hid the action
buttons), https://wizardofodds.com/play/blackjack-v2/, 6 decks / 75–80% penetration,
short 10–15 min sessions, **split every pair**, capture at least once at each
round's settle. Run `dataset_report.py` after every session.

**Auto-label:** corner-strip template matching (`vision/autolabel.py`); back card
uses the FULL sprite (its corner under-scores after resampling). Class order =
`CLASS_NAMES`: 13 ranks (2..9,T,J,Q,K,A) × suits c,d,h,s + `back` = 53.

**Synthetic pool:** deficit-driven planner; regeneration recipe (seed/targets) is in
`data/synthetic/synth_meta.json`. Accepted gap: no chips/buttons/rule-text on
synthetic backgrounds — revisit ONLY if M4 val shows false positives.

**Colab run (user):** upload `notebooks/train_yolo.ipynb`, T4 runtime, upload the
zip, run all cells, unzip `bjcounter_weights.zip` into `models/`. Augments must stay
geometry-preserving (NO flips/rotations — card corners are orientation-specific);
ONNX exports pinned to opset 17. Record provenance hashes in `models/README.md`.

**M4 gates** (checked by `eval_detector.py` with OUR pre/post-processing, not
Colab's numbers): mAP50 ≥ 0.99, per-rank recall ≥ 0.995 at conf 0.80, <100 ms/frame
CPU median — latency is timed on a deployed-conf detector, never the AP sweep.

## 6. Tracker rules that are easy to get wrong (learned the hard way)

- The §4.2 flicker gate applies to NON-deal-shaped frames only, and the player zone
  is exempt when hand count increased (splits deal replacements onto old fan
  positions). §14 M5 rows.
- Deal-shaped frames pass a **two-tier persistence rule**: previous frame also
  deal-shaped → ANY persisting face refuses NEW_DEAL (a double-flicker recapture
  silently double-counted under a lone ≥2 bar — review-verified CRITICAL);
  otherwise ≥2 refuses, exactly 1 stays NEW_DEAL (~2%/round exact-repeat
  coincidence that must recount or rounds silently vanish).
- Fresh shoe accepts only a deal-shaped frame; everything else is SUSPECT.
- A hand's leftmost card must sit at its layout anchor (fan slot 0) — this pins the
  NEW_DEAL anchor check to `len(player_hands) == 1`.
- Per-class NMS at IoU 0.6: same-class fan neighbours sit at ~0.47 (must survive),
  duplicate boxes at ~0.9 (must not).
- Detector conf default 0.80 == `tracker/state.py CONF_THRESHOLD`. Keep in sync.

## 7. Trainer facts (from source-level research — trust these over intuition)

- Canvas is a fixed 960×640 DOM (`table.png`), cards are 67×94 CSS sprites from
  `deck.png` (back at y=5358), pixel-identical every deal.
- Dealer fan anchors at (422,10), step +24,0. Player layouts per hand count:
  1:(443,416) · 2:+(638,377) · 3:(196,358),(443,416),(638,377) ·
  4:(50,250),(280,395),(500,415),(675,345); player fan step +24,−4.
- Captures include the count bar (36 CSS px × scale) ABOVE the felt —
  `table_origin=(0, bar)` when assembling TableStates from captured frames.
- Trainer's displayed count only reflects revealed cards (hole folds in on reveal) —
  RC is compared EXACTLY in evals; TC/DecksLeft need normalization (trainer's
  formula has a +1 and counts dealt-unrevealed cards).
- **Never parse the dealer badge** — it leaks the hole-inclusive total
  (UpdateLabel called without uponly).
- Silent reshuffle at cut card; only cue is the "LAST HAND" badge at (850,4).
- Split-count quirk (`Shoe.RunningCount += CountValue[10]` per split) is STILL
  UNVERIFIED live — user must observe RC before/after a split once; the eval
  harness normalizes via SPLIT events if real. Never mirror it in our count.

## 8. Security profile (checked at Phase 7 — keep true continuously)

- `pynput` registers named hotkeys ONLY. No key logging/buffering, ever.
- Zero network egress in `src/` (the only egress in the whole project is the Colab
  notebook: labeled crops up, weights down). Raw screenshots stay in gitignored
  `data/raw/` and never leave the machine.
- Deps pinned in `pyproject.toml`; checkpoint provenance (URL + SHA-256) in
  `models/README.md`; `.claude/settings.json` deny baseline stays intact.

## 9. Human-in-the-loop registry (things ONLY the user can do)

1. Capture sessions at the trainer (val/test variety; splits).
2. The split-count quirk observation (§7).
3. Run the Colab notebook; place weights in `models/`.
4. Record 3–5 real shoes with hand-labeled counts → `tests/fixtures/shoes/`
   (M5 exit gate; easiest once M6's app exists).
5. Be present for the M6 live smoke test and the Phase 8 20-shoe eval.
6. Approve the Phase 9 ship/README push.

## 10. Roadmap from here

M4 (user Colab run → `eval_detector.py` gates) → M6 app shell (`app/`: config,
hotkeys, worker thread, tkinter overlay — threading model pinned in ARCHITECTURE §7,
including the QUIT-sentinel shutdown order) → M5 exit (recorded-shoe replay) →
RL chapter (`rl/`: env → tabular MC → divergence report vs published indices,
honest matches AND mismatches) → Phase 6–9 (full-repo reviews incl.
silent-failure-hunter on tracker/vision, security scan, 20-shoe pass^k eval,
README + demo GIF, ship).
