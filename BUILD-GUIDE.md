# BUILD-GUIDE — blackjack-counter

> Generated 2026-07-02 by `/personalized`. This file is a **runnable build prompt**: open this
> folder as a Claude Code project, have Claude read this file, and execute the phases in order.
> Each phase names the exact ECC skill/command, agent, model tier, inputs, outputs, and exit
> criteria. Do not skip phases. Do not start a phase until the previous phase's exit criteria pass.

---

## 1. Objective

**blackjack-counter** — a real-time computer-vision card-counting advisor, built as an AI-engineer
portfolio project. On a global hotkey, it screenshots the blackjack table region, detects every
visible card with a fine-tuned YOLO model, reconciles table state across frames, maintains a Hi-Lo
running count and true count, and displays the optimal action (hit / stand / double / split /
surrender) from basic strategy plus Illustrious 18 + Fab 4 count deviations in an always-on-top
overlay.

**Target surface:** the [Wizard of Odds free blackjack trainer](https://wizardofodds.com/play/blackjack-v2/)
— chosen because it has fixed digital card art (ideal for detection), configurable decks and
**deck penetration** (set 6 decks, 75–80% penetration so true counts actually swing), and a
built-in count display that serves as **ground truth** for end-to-end accuracy evals.

**v1 scope includes the RL chapter:** train a tabular RL agent on a custom count-aware blackjack
environment and show it rediscovers basic strategy and the count deviations — benchmarked against
the deterministic table. The lookup table is ground truth; the RL agent is the portfolio story.

**Ethics/framing constraint (non-negotiable, goes in README):** this is a training/practice tool
for free-play simulators. It is not for real-money play — real-time assistance violates casino
ToS everywhere and device statutes in some jurisdictions. Counting is also mathematically dead on
RNG blackjack (per-hand shuffle) and neutered on live-dealer tables (~50% penetration, 20–50
rounds/hr, ~$1/hr EV on a $40k bankroll per published sims). The README says all of this plainly.

### Assumptions (flag deviations, don't silently absorb them)

- Host: Windows 11, Python 3.11+. No local NVIDIA GPU — **YOLO fine-tuning runs on free Colab T4**;
  everything else (inference included) runs local CPU.
- Trainer config for all evals: 6 decks, deepest available penetration, note H17/S17 + DAS
  settings used and match the strategy tables to them.
- Single-player single-seat play (the WoO trainer is single-seat; multi-hand appears only via splits).
- Repo will be published to GitHub (advayiyer7) once Phase 8 gates pass.

---

## 2. Stack

| Concern | Choice |
|---|---|
| Language / runtime | Python 3.11+, `venv` + `pip` (or `uv` if preferred), `pyproject.toml` |
| Screen capture | `mss` (fast region-of-interest grabs) |
| Global hotkeys | `pynput` (registered hotkeys ONLY — see security profile) |
| Card detection | Ultralytics YOLOv8n (or v11n), fine-tuned on WoO card art; ONNX/CPU inference |
| Auto-labeling | OpenCV template matching against extracted card sprites |
| Overlay UI | `tkinter` always-on-top frameless window (stdlib, zero deps) |
| RL chapter | `gymnasium`-style custom env + `numpy` tabular Monte Carlo control / Q-learning |
| Tests / quality | `pytest` + `pytest-cov` (≥80%), `ruff` (format + lint) |
| Training compute | Google Colab free T4 — `notebooks/train_yolo.ipynb`; weights exported to `models/` |

**File size discipline:** 200–400 lines typical, 800 hard max. Small cohesive modules.

## 3. Repo layout (create in Phase 0)

```
blackjack-counter/
├── BUILD-GUIDE.md              # this file
├── pyproject.toml
├── README.md
├── .claude/settings.json       # deny baseline — Phase 0
├── src/bjcounter/
│   ├── strategy/               # tables.py, deviations.py, engine.py   (pure, no I/O)
│   ├── counting/               # hilo.py, true_count.py, shoe.py       (pure, no I/O)
│   ├── vision/                 # capture.py, detector.py, autolabel.py
│   ├── tracker/                # state.py, reconcile.py, rounds.py
│   ├── rl/                     # env.py, train.py, compare.py
│   └── app/                    # hotkeys.py, overlay.py, main.py
├── tests/                      # mirrors src; fixtures/ holds recorded frame sequences
├── data/                       # datasets (raw screenshots gitignored; YOLO labels kept)
├── models/                     # best.pt / best.onnx + training metrics json
├── notebooks/                  # train_yolo.ipynb (Colab)
└── docs/                       # PRD.md, ARCHITECTURE.md, EVAL.md, RL-REPORT.md, research/
```

## 4. Security profile — moderate, with two specific hot spots

Foreign content in play: pixels/screenshots from a third-party website, a pretrained YOLO
checkpoint, and a Colab boundary. No secrets, no payments, no user data. The minimum bar plus two
project-specific rules:

1. **The hotkey listener must never be a keylogger.** `pynput` registers the app's specific
   hotkeys (e.g. capture, reset-count, quit) and nothing else. No global key event logging, no
   buffering of keystrokes, ever. Code review explicitly checks this.
2. **Screenshots stay local and narrow.** Capture only the user-selected game-window region, not
   the full screen. Raw screenshots live in `data/raw/` (gitignored). The runtime app makes **zero
   network calls** — inference is fully offline. The only network egress in the whole project is
   the Colab notebook (uploads: labeled card crops only; downloads: weight files only).

`.claude/settings.json` deny baseline (write in Phase 0):

```json
{
  "permissions": {
    "deny": [
      "Read(~/.ssh/**)", "Read(~/.aws/**)", "Read(**/.env*)",
      "Bash(curl * | bash)", "Bash(ssh *)", "Bash(scp *)", "Bash(nc *)"
    ]
  }
}
```

Supply chain: pin `ultralytics`, `torch`, `mss`, `pynput` versions in `pyproject.toml`; run
`/security-scan` over the project's own hooks/skills/MCP surface before trusting them; treat the
downloaded pretrained checkpoint as an artifact (record its source URL + hash in `models/README.md`).

---

## 5. Execution pipeline

`RESEARCH → PRD → ARCHITECTURE → TDD IMPLEMENT (M1–M6) → RL CHAPTER → REVIEW → SECURITY → VERIFY → SHIP`

Run `/save-session` at the end of each phase; `/clear` (or compact) between heavy phases. All
intermediate outputs go to files under `docs/` — never leave load-bearing results only in chat.

### Phase 0 — SCAFFOLD (main instance · Sonnet)

Create the repo layout above: `git init`, `pyproject.toml` (deps pinned, ruff + pytest config),
`.claude/settings.json` deny baseline, `.gitignore` (`data/raw/`, `*.pt` optional via LFS),
README skeleton with the ethics/framing paragraph. **Exit:** `pytest` runs (0 tests OK), `ruff
check` clean, initial commit made.

### Phase 1 — RESEARCH (3 subagents in parallel · Haiku/Sonnet · Explore agent)

Fan out three **parallel** research subagents; main instance idles until all return:

| # | Task | Skill/agent | Output |
|---|---|---|---|
| R1 | WoO trainer mechanics: exact URL/settings for decks + penetration, card art appearance, layout regions (player/dealer/count display), how splits render, shuffle indication | Explore subagent + `WebFetch` | `docs/research/trainer-notes.md` |
| R2 | Stack notes: ultralytics fine-tune + ONNX export API (current version), `mss` region capture on Windows, `pynput` hotkey registration, tkinter always-on-top overlay | `documentation-lookup` | `docs/research/stack-notes.md` |
| R3 | **Canonical strategy tables**: 6-deck basic strategy for the trainer's exact rules (H17/S17, DAS, surrender availability), Hi-Lo tag values, true-count conversion, Illustrious 18 + Fab 4 deviation indices — cross-checked from ≥2 authoritative sources (Wizard of Odds strategy pages, Schlesinger's published I18) | Explore subagent + `WebSearch` | `docs/research/strategy-tables.md` — **machine-readable tables (CSV/JSON blocks), these become test fixtures** |

**Exit:** all three files exist; R3 tables agree across both sources cell-for-cell (discrepancies resolved and noted).

### Phase 2 — PRD (main · planner agent · Sonnet)

Run `/prp-prd` with the objective + research notes. Keep it lean: user story (Advay practicing
counting), functional requirements per module, non-goals (no real-money use, no multi-site support
in v1, no automated bet sizing UI beyond a TC-indexed bet suggestion), success metrics (the Phase 8
gates). **Output:** `docs/PRD.md`. **Exit:** PRD lists measurable success criteria matching §Phase 8.

### Phase 3 — ARCHITECTURE (main · architect agent · **Opus**)

Run `/plan` (architect + code-architect) reading `docs/PRD.md` + all research notes. Must decide
and document:

- Dataclass contracts: `Card(rank, suit, bbox, conf)`, `TableState(player_hands, dealer_cards, frame_id)`,
  `RoundState`, `ShoeState(cards_seen, decks_total)` — **frozen/immutable dataclasses**, new state per frame.
- The frame-reconciliation algorithm (how `tracker/reconcile.py` diffs consecutive `TableState`s to
  emit only newly-revealed cards; how round boundaries are detected; how splits map to multiple hands).
- Detector interface so the model is swappable (template-matcher fallback implements the same protocol —
  useful before the YOLO weights exist).
- Pipeline wiring: hotkey → capture → detect → reconcile → count → decide → overlay, all pure
  functions around an explicit state object; side effects only at the edges (`app/`).

**Output:** `docs/ARCHITECTURE.md`. **Exit:** user has reviewed and CONFIRMED the plan (plan skill waits).

### Phase 4 — TDD IMPLEMENT (main · tdd-guide · Sonnet; Haiku for mechanical scripts)

Strict RED→GREEN→REFACTOR per milestone. After **each** milestone: run `/code-review` +
`/python-review` (parallel subagents), fix CRITICAL/HIGH, then `/checkpoint` commit
(`feat: <milestone>` conventional commits).

**M1 — Strategy engine** (`strategy/`). Tests are *generated from* `docs/research/strategy-tables.md`:
every chart cell = one test case (~340 hand/upcard combos + deviation overrides at their TC indices).
Exit: 100% chart parity, module coverage ≥95%, zero I/O in module.

**M2 — Counting engine + shoe simulator** (`counting/`). Hi-Lo tags, running count, true count =
RC / decks-remaining (decks-remaining from cards_seen), shuffle reset. Property tests: a full shoe
always sums to RC 0; TC sign matches RC sign; simulator deals 10k shoes with zero count drift.
Exit: property tests pass, coverage ≥95%. *(M1+M2 need no vision — the whole logic core is provable
before a single pixel is read.)*

**M3 — Dataset pipeline** (`vision/capture.py`, `vision/autolabel.py`; Haiku subagent OK for
script scaffolding). Capture script grabs region screenshots at the WoO trainer while playing
manually (~30–60 min of play, varied hands/splits). Auto-labeler template-matches the 52 card
sprites (extract sprites once from screenshots) → YOLO-format labels. Hand-verify a 10% sample.
Target ~1,500–2,500 labeled frames, all 52 classes ≥40 instances (rank is what matters; keep suit
for detection robustness, count uses rank only). Exit: dataset lint passes (label/image parity,
class-balance report in `data/REPORT.md`), sample verification error <1%.

**M4 — Detector fine-tune (Colab)** (`notebooks/train_yolo.ipynb` + `vision/detector.py`).
Fine-tune YOLOv8n on the M3 dataset (T4, ~15–30 min), export `best.pt` + ONNX, commit weights +
`models/metrics.json`. Local eval script on a held-out split. If torch/CUDA issues: dispatch
**pytorch-build-resolver** subagent. Exit: **mAP50 ≥0.99 and per-rank recall ≥0.995 on held-out
frames** (fixed digital art should be near-perfect — if it isn't, the dataset is wrong, not the
model); CPU inference <100ms/frame via ONNX.

**M5 — State tracker** (`tracker/`). The genuinely hard part. Record 3–5 full shoes as frame
sequences (`tests/fixtures/shoes/`) with hand-labeled expected counts. Implement reconciliation
(new-card diffing keyed by position+rank), round-boundary detection (table cleared), split
handling, dealer hole-card reveal, shuffle reset (manual hotkey in v1; auto-detect is a stretch).
Exit: replaying every recorded shoe reproduces the hand-labeled running count **exactly** at every
frame; coverage ≥85%.

**M6 — App shell** (`app/`). Region selector (drag-once, persisted to config), hotkeys (capture /
reset-count / quit), tkinter overlay showing RC, TC, decks remaining, recommended action + whether
it's a deviation, TC-indexed bet suggestion. Exit: live smoke test at the WoO trainer — 2 full
shoes, overlay updates <300ms after hotkey, no crashes across splits/blackjacks/pushes.

### Phase 5 — RL CHAPTER (main · Sonnet; review by **mle-reviewer** subagent)

`rl/env.py`: count-aware blackjack env matching the trainer's rules — state =
(player_total, usable_ace, pair_rank_or_none, dealer_upcard, TC_bucket ∈ {≤−2,−1,0,+1,+2,+3,≥+4}),
actions = {hit, stand, double, split, surrender if available}, dealt from a simulated 6-deck shoe
at the same penetration so TC states occur naturally. `rl/train.py`: tabular first-visit Monte
Carlo control (or Q-learning) — expect ~20–50M hands, hours on CPU / free Colab, checkpoint the
Q-table. `rl/compare.py`: policy-vs-M1-table divergence heatmaps per TC bucket + EV comparison via
the M2 simulator.

**Output:** `docs/RL-REPORT.md` — the portfolio writeup: "agent rediscovers basic strategy;
deviations emerge at the published I18 indices (16v10 stands at TC≥0, 15v10 at TC≥+4, insurance
at TC≥+3, …)". **Exit:** ≥97% action agreement with basic strategy in the TC=0 bucket; each
detected deviation compared honestly against its published index (matches AND mismatches reported —
honest findings are the differentiator, same as the semantic-cache project).

### Phase 6 — REVIEW (parallel subagents · Sonnet)

Full-repo pass, three subagents in parallel: `/code-review` (code-reviewer), `/python-review`
(python-reviewer), **silent-failure-hunter** focused on `tracker/` + `vision/` (dropped detections,
swallowed reconciliation errors — a silently wrong count is this app's worst failure mode; the
overlay must surface low-confidence frames instead of guessing). Plus **mle-reviewer** on
`rl/` + training/eval code if not already done in Phase 5. **Exit:** zero CRITICAL/HIGH open.

### Phase 7 — SECURITY (security-reviewer agent · **Opus**)

`/security-scan` (AgentShield) over the project's hooks/skills/settings + `/security-review` of the
code. Focus list: hotkey listener scope (rule §4.1), screenshot data locality (rule §4.2), zero
runtime egress (grep for network imports in `src/`), pinned deps, no secrets, checkpoint provenance
recorded. **Exit:** no CRITICAL/HIGH; findings + dispositions appended to `docs/EVAL.md`.

### Phase 8 — VERIFY (main · Sonnet · e2e-runner mindset)

1. `/test-coverage` — overall ≥80% (strategy/counting will carry ≥95%).
2. **End-to-end eval — pass^k, not pass@k** (the tool must be right *every* time, not once):
   20 full shoes at the WoO trainer with its built-in count display ON; log predicted vs actual RC
   each round. **Gates: zero count drift in ≥19/20 shoes (any drifting shoe gets a root-cause note),
   card-detection accuracy ≥99.5%, decision matches the M1 table 100% given the tracked state.**
3. `/verify` for the app-launch smoke path. Results → `docs/EVAL.md` (the portfolio metrics table).

### Phase 9 — SHIP (main · Sonnet)

README final: demo GIF (hotkey → overlay flow), architecture diagram (Mermaid), EVAL metrics table,
RL-chapter summary + link, honest-limitations section (single-site card art, manual shuffle reset,
why online real-money counting is dead anyway — cite the research), ethics note. Push to GitHub.
Optional later: project page in the `testportfolio` site (separate session, that repo).

---

## 6. Parallelization map — 1 main instance + subagents (no worktrees)

Chosen deliberately: milestones are sequential by dependency (M5 needs M4's weights; M6 needs M5),
so worktrees would idle. Parallelism lives inside phases:

- Phase 1: R1/R2/R3 subagents **in one message** (parallel).
- Phase 4: while main does M2 (pure logic), a Haiku subagent may scaffold M3's capture/label scripts —
  disjoint files (`counting/` vs `vision/`), safe.
- M4 training runs on Colab **in the background** while main starts M5 against the
  template-matcher fallback detector (same interface — this is why Phase 3 defines the protocol).
- Phases 6: review subagents in parallel.

## 7. Token budget plan

| Work | Model |
|---|---|
| Research fan-out, dataset/glue scripts, dataset lint | Haiku |
| Main implementation, TDD, reviews, RL chapter | Sonnet |
| Phase 3 architecture, Phase 7 security review | Opus |

Run `/model-route` when unsure; `/context-budget` before Phase 4 and Phase 5; `/save-session` at
each phase end and `/resume-session` to continue cold; keep chat lean by writing results to
`docs/` and referencing paths. Fixtures > prose: strategy tables and recorded shoes live as data
files, not conversation content.

## 8. Security checklist (tick at Phase 7)

- [ ] `pynput` registers named hotkeys only — no keystroke logging/buffering anywhere
- [ ] Capture is region-scoped; raw screenshots gitignored; nothing uploaded except Colab card crops
- [ ] `src/` has zero network egress (imports audited)
- [ ] `.claude/settings.json` deny baseline present
- [ ] Deps pinned; pretrained checkpoint source URL + hash recorded
- [ ] `/security-scan` clean over hooks/skills/MCP
- [ ] No secrets anywhere (there should be none in this project at all)
- [ ] README ethics/ToS framing present

## 9. Session-memory plan

End of each phase: `/save-session`. Update the project memory
(`ai-engineer-portfolio-projects.md`) at three moments: after Phase 3 (architecture locked), after
Phase 8 (eval numbers — record the actual metrics), after Phase 9 (shipped + repo URL).

## 10. Handoff prompt

Start a fresh Claude Code session **in `C:\Users\chris\Desktop\blackjack-counter`** and paste:

> Read BUILD-GUIDE.md in full. Execute it phase by phase starting at Phase 0. Do not skip phases
> or exit criteria. Phase 1 research subagents run in parallel. Stop for my confirmation only at
> the Phase 3 architecture review and before the Phase 9 GitHub push. When a phase's exit criteria
> pass, /save-session, then continue.
