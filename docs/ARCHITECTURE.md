# ARCHITECTURE — blackjack-counter

> BUILD-GUIDE Phase 3 output. Inputs: `docs/PRD.md`, `docs/research/*`. Reviewed adversarially by
> an architect pass; findings 1–13 incorporated (see §13). **Status: CONFIRMED by user
> 2026-07-03.** Everything here is a decision, not a suggestion — deviations during
> implementation must be flagged and recorded here.

## 1. Design principles

1. **Pure core, effectful edges.** `strategy/`, `counting/`, `tracker/` are pure functions over
   frozen dataclasses — no I/O, no globals, no clocks. All side effects (screen, keyboard, UI,
   model file loading) live in `vision/` and `app/`.
2. **Immutable state.** Every state object is a frozen dataclass; each frame produces *new* state.
   No in-place mutation anywhere in the core.
3. **Never guess silently.** Any frame that can't be interpreted with confidence updates *nothing*
   and surfaces a warning in the overlay. A skipped frame is recoverable (recapture); a silently
   wrong count is the worst failure mode (PRD risk).
4. **Position-free count arithmetic; position-aware boundaries.** The count itself derives from
   table-wide multiset deltas (immune to split relocation and skipped frames). But *round
   boundaries* are detected from positive structural signals — deal shape, hole-card transitions,
   bbox continuity — never inferred from multiset mismatch alone. A mismatch without a positive
   boundary signal is a SUSPECT frame that changes nothing. This split is the load-bearing
   architectural idea (§4), and the second half of it exists because the review proved that
   multiset-only boundary detection silently corrupts the count (§13 findings 1–3).

## 2. Module map and dependency rules

```
src/bjcounter/
├── types.py          # shared frozen dataclasses + enums; stdlib only, imported by everyone
├── strategy/         # pure, stdlib only
│   ├── tables.py     #   basic-strategy tables (H17 + S17 diff), data only
│   ├── deviations.py #   I18 + Fab 4 entries, rules-aware filtering, data + tiny logic
│   ├── hands.py      #   hand math: total, softness, pair detection, legality (ctx-aware)
│   └── engine.py     #   decide(hand, upcard, tc, rules, ctx) -> Advice
├── counting/         # pure, stdlib only
│   ├── hilo.py       #   TAGS: Mapping[Rank, int]; tag arithmetic
│   ├── true_count.py #   decks_remaining, floored TC (formulas pinned in §5)
│   └── shoe.py       #   ShoeState + seeded shoe simulator (also reused by rl/)
├── tracker/          # pure; consumes types.py only (no numpy, no counting import)
│   ├── state.py      #   TableState construction from detections; zone/hand assignment
│   ├── reconcile.py  #   transition classifier + frame diff (the algorithm, §4)
│   └── rounds.py     #   RoundState, settledness rules
├── vision/           # effectful; numpy/cv2/onnxruntime/mss
│   ├── capture.py    #   mss region grab -> np.ndarray (BGR)
│   ├── detector.py   #   CardDetector protocol + OnnxYoloDetector + TemplateMatchDetector
│   └── autolabel.py  #   M3 dataset tooling (sprite templates -> YOLO labels)
├── rl/               # numpy; depends on counting/ + strategy/ only
│   ├── env.py        #   count-aware blackjack env (gymnasium-style)
│   ├── train.py      #   tabular MC control / Q-learning
│   └── compare.py    #   policy vs strategy-table divergence + EV comparison
└── app/              # effectful edges; tkinter/pynput
    ├── main.py       #   entrypoint: DPI awareness FIRST, wiring, threads, shutdown
    ├── config.py     #   frozen AppConfig load/save (region, rules, thresholds)
    ├── hotkeys.py    #   pynput GlobalHotKeys (named combos ONLY — security rule §4.1)
    ├── worker.py     #   single worker thread: capture->detect->reconcile->decide
    └── overlay.py    #   tkinter always-on-top overlay, queue-fed via root.after
```

Dependency direction (arrows = "imports from"):
`app → {vision, tracker, counting, strategy, types}`; `vision → types`; `tracker → types`;
`rl → {counting, strategy, types}`; `strategy/counting → types` only. **`tracker` does NOT import
`counting`** — it emits revealed ranks; `app/worker.py` applies them to `ShoeState`. This keeps
both independently testable and the count pipeline explicit in one place.

Additions vs the BUILD-GUIDE §3 sketch (flagged per its own rule): `types.py`,
`strategy/hands.py`, `app/config.py`, `app/worker.py`. All small, all within the file-size
discipline; the guide's layout listed representative files, not an exhaustive set.

## 3. Data contracts (`types.py` unless noted)

All `@dataclass(frozen=True, slots=True)`. Enums are `StrEnum` unless noted.

```python
class Rank(StrEnum):   # "2".."9", "T", "J", "Q", "K", "A"
class Suit(StrEnum):   # "s", "h", "d", "c"
class Action(StrEnum): # HIT, STAND, DOUBLE, SPLIT, SURRENDER
class Surrender(StrEnum):  # NONE, ANY_CARD, NOT_VS_ACE
class DoubleRule(StrEnum): # ANY_TWO, HARD_9_TO_11, HARD_10_11

class Event(StrEnum):
    NEW_ROUND = "new_round"          # positive deal signature accepted
    HOLE_REVEALED = "hole_revealed"  # dealer back -> face this frame
    SPLIT = "split"                  # player hand count increased (eval normalization hook)
    ROUND_ENDED = "round_ended"      # cards removed w/o new deal (surrender / table clear)
    DUPLICATE_FRAME = "duplicate"    # identical to previous accepted frame (double-tap)
    PREV_ROUND_UNSETTLED = "prev_round_unsettled"  # new deal arrived but last round never
                                     # reached a settled-looking frame -> count may be short

BBox = tuple[int, int, int, int]     # x, y, w, h — capture-region pixel coords

@dataclass(frozen=True, slots=True)
class Card:
    rank: Rank
    suit: Suit
    bbox: BBox
    conf: float

@dataclass(frozen=True, slots=True)
class Detection:                     # raw detector output; "back" has no rank
    label: str                       # "2s".."Ah" or "back"
    bbox: BBox
    conf: float

@dataclass(frozen=True, slots=True)
class Hand:
    slot: int                        # trainer hand slot 0..3
    cards: tuple[Card, ...]          # left-to-right fan order

@dataclass(frozen=True, slots=True)
class TableState:
    frame_id: int
    player_hands: tuple[Hand, ...]
    dealer_cards: tuple[Card, ...]   # fan order; face-down hole = NOT in this tuple
    dealer_has_hole: bool            # a "back" detection sits in the dealer zone
    last_hand: bool = False          # "LAST HAND" badge detected (stretch; default off)
    warnings: tuple[str, ...] = ()

@dataclass(frozen=True, slots=True)
class Rules:                         # trainer config mirror; defaults = PRD reference config
    decks: int = 6                   # NB: the trainer's own default is 8 — must be set to 6
    h17: bool = True
    das: bool = True
    surrender: Surrender = Surrender.ANY_CARD
    peek: bool = True
    double: DoubleRule = DoubleRule.ANY_TWO
    max_hands: int = 4               # SplitX "up to 3 times" -> 4 hands
    max_ace_hands: int = 2           # SplitA "once" -> 2 hands
    hit_split_aces: bool = False

# counting/shoe.py
@dataclass(frozen=True, slots=True)
class ShoeState:
    seen: tuple[Rank, ...]           # every counted rank this shoe, in reveal order
    decks_total: int = 6
    # derived properties: cards_seen, running_count, decks_remaining, true_count — formulas §5

# tracker/rounds.py
@dataclass(frozen=True, slots=True)
class RoundState:
    round_index: int
    table: TableState | None         # last ACCEPTED frame (None = fresh shoe)
    counted: tuple[tuple[Rank, Suit], ...]  # sorted multiset of faces counted this round
    hole_was_revealed: bool          # this round reached a hole reveal
    settled: bool                    # round reached a settled-looking frame (§4.5)

@dataclass(frozen=True, slots=True)
class ReconcileResult:
    round_state: RoundState
    revealed: tuple[Card, ...]       # newly visible face-up cards -> count these
    events: tuple[Event, ...]
    accepted: bool                   # False = SUSPECT frame, nothing changed
    warnings: tuple[str, ...]

# strategy/engine.py
@dataclass(frozen=True, slots=True)
class TableContext:                  # what a single HandView can't know (§13 finding 6)
    num_hands: int                   # 1 = no split yet
    # (amended in M1: per-hand card count is NOT carried here — decide() receives the
    # hand's cards directly and derives it in HandView.n_cards; see §14)

@dataclass(frozen=True, slots=True)
class Advice:
    action: Action | None            # None = no legal decision to display
    fallback: Action | None          # e.g. D falls back to H when doubling illegal
    is_deviation: bool
    deviation: str | None            # e.g. "I18: 16vT stand @ TC>=0"
    insurance: bool                  # dealer shows A and TC >= +3
    caveat: str | None               # e.g. "3+ cards: ignore if hand already doubled" (§6)

# app/worker.py -> app/overlay.py payload
@dataclass(frozen=True, slots=True)
class OverlayModel:
    running_count: int
    true_count: int                  # floored
    decks_remaining: float
    per_hand: tuple[tuple[int, Advice], ...]   # (slot, advice), all visible unresolved hands
    bet_units: int
    events: tuple[Event, ...]
    warnings: tuple[str, ...]
    last_hand: bool
```

Rationale for `ShoeState.seen` as a full rank tuple rather than an int pair: ≤312 entries,
enables recomputation/validation in tests, gives the RL env and the eval harness the exact same
substrate, and keeps `running_count`/`true_count` as derived pure properties (single source of
truth, no drift between fields).

## 4. Frame reconciliation (`tracker/`) — the load-bearing algorithm

### 4.1 Two channels, deliberately separated

- **Counting channel (drives the count; must be near-perfect):** table-wide **multiset delta** of
  face-up `(rank, suit)` pairs — but only after the transition classifier (§4.2) has positively
  established which of {duplicate, continuation, round end, new deal} this frame is. Position-free
  arithmetic; position-aware classification.
- **Decision channel (drives displayed advice; wrong = annoying, not corrupting):** zone/hand
  assignment of the *current frame only* (§4.3). Advice is recomputed per frame from the visible
  hands; it never feeds back into the count.

### 4.2 Transition classifier + reconcile step (pure function)

```
step(round_state, table_now) -> ReconcileResult
  0. GATE: any detection conf < threshold (default 0.80), ambiguous zone assignment, or a
     label CHANGE at a continuing bbox position (suit/rank flicker)
       -> accepted=False, SUSPECT, state unchanged, warn "recapture".
  1. faces_now = multiset of (rank, suit) of all face-up cards ("back" excluded)
  2. Classify the transition — first match wins:
     DUPLICATE     faces, positions, and hole flag identical to round_state.table
                   -> no-op; emit DUPLICATE_FRAME (hotkey double-tap; see residual risk §12).
     NEW_DEAL      positive deal signature: exactly 1 player hand of exactly 2 cards at the
                   1-hand deal anchor, dealer showing exactly 1 face + hole present.
                   (Two consecutive deal-shaped frames with ANY difference = two rounds.)
     CONTINUATION  faces_now ⊇ round_state.counted (multiset), AND bbox continuity holds:
                   dealer fan may only append; player positions persist unless hand count
                   increased (SPLIT relocation is legal, then multiset continuity governs).
     ROUND_ENDED   cards removed with NO new faces (player zone cleared and/or table cleared;
                   dealer upcard may persist at its old position; not deal-shaped)
                   -> revealed = ∅, counted frozen, settled=True. (Covers surrender + clears.)
     SUSPECT       anything else — e.g. a card vanished while others continue (dropped
                   detection), partial overlap of old/new cards, superset holds but dealer
                   fan shrank -> accepted=False, nothing changes, warn.
  3. On NEW_DEAL:  if previous round existed and NOT round_state.settled
                     -> emit PREV_ROUND_UNSETTLED (count may be short; overlay shows it).
                   counted' = faces_now; revealed = all faces_now; round_index += 1.
     On CONTINUATION: revealed = faces_now − counted; counted' = faces_now;
                   emit HOLE_REVEALED / SPLIT when those transitions are observed.
  4. Update settledness (§4.5); return ReconcileResult.
```

Consequences worth stating:

- **A dropped detection can no longer fake a round boundary.** Losing one card of a continuing
  round fails CONTINUATION (not a superset) *and* fails NEW_DEAL (wrong shape) → SUSPECT, warn,
  recapture. The count is untouched. (Review finding 1 — previously this recounted every
  surviving card.)
- **Surrender is safe.** The post-surrender frame (player cleared, dealer upcard persists, hole
  down) classifies as ROUND_ENDED: nothing is recounted. (Finding 2 — previously the dealer
  upcard was double-counted on every captured surrender, a systematic error.)
- **One capture at settle is sufficient for a correct count** — at settle every revealed card is
  face-up and the CONTINUATION delta picks up everything uncounted. Mid-round captures are only
  needed for advice.
- **Hole-card reveal and splits need no special case in the count** — the revealed rank appears
  in the delta; relocated split cards are already in `counted`. Both still emit their events
  (HOLE_REVEALED, SPLIT) because the eval harness needs per-round split counts to normalize the
  trainer's split-count quirk (finding 7; trainer-notes §9.7).
- **Suit misreads can't fake a boundary** (finding 4): a label change at a continuing position is
  gated at step 0; count integrity does not depend on per-suit recall (which is ungated — only
  per-rank recall has a gate).

### 4.3 Zone / hand assignment (`tracker/state.py`)

Trainer geometry is fixed (research §4): dealer fan starts (422,10); player hand origins per
hand-count are exact; fan steps are (+24, 0) dealer / (+24, −4) player; card sprite is 67×94 at
scale 1.0.

- **Self-calibration:** `scale = median(detected card width) / 67` per frame — no dependence on
  exact region-drag precision or the 0.9/0.74 CSS breakpoints.
- Anchor sets for 1/2/3/4-hand layouts are scaled; best-fitting layout chosen by total assignment
  distance; cards join the nearest hand anchor-line, ordered by x within a hand.
- A card too far from every anchor (> ~1 card-width) ⇒ ambiguous ⇒ SUSPECT gate (§4.2.0).
- **Cardinality sanity check** (finding 12): a hand's card count must match its fan extent
  (`(rightmost−leftmost)/(24·scale) + 1`); mismatch ⇒ SUSPECT — this catches NMS suppressing one
  card of a tight fan *and* duplicate boxes, independent of detector confidence.
- Dealer upcard = leftmost dealer face; `dealer_has_hole` = any "back" in the dealer zone.

### 4.4 Shuffle handling

Manual: reset hotkey → fresh `ShoeState`/`RoundState` (v1 contract, per PRD). Stretch (already
plumbed via `TableState.last_hand`): the "LAST HAND" badge sits at fixed coords (850,4) — a cheap
template match in `vision/` sets the flag; overlay shows "LAST HAND — reset after this round".
Not a gate for any milestone.

### 4.5 Settledness and the user protocol (finding 5)

A round is **settled** when any holds: the hole was revealed this round; the player zone cleared
(surrender/clear); or the sole player hand is a 2-card natural (A + ten-value). If a NEW_DEAL
arrives while the previous round is unsettled, the tracker emits PREV_ROUND_UNSETTLED and the
overlay states plainly that the count may be short — it cannot be reconstructed, only flagged.

**User protocol (amends PRD user-flow step 2; PRD updated):** capture *at least once at each
round's settle* — that capture alone keeps the count exact. Capture additionally at decision
points whenever advice is wanted. The overlay's unsettled warning is the backstop when the settle
capture is forgotten.

## 5. Counting formulas (pinned; finding 8; amended in M2 — see §14)

```
running_count   = Σ HILO_TAG[rank] over ShoeState.seen
remaining_cards = max(decks_total·52 − len(seen), 26)         # 26 = half-deck clamp
true_count      = (running_count · 52) // remaining_cards     # EXACT integer floor-division
decks_remaining = remaining_cards-style float                 # DISPLAY ONLY (overlay)
```

TC must be computed in exact integer arithmetic: an IEEE-754 intermediate like 60/52 puts
mathematically-integer TCs one ULP below the floor boundary (RC −15 with 60 cards left is
exactly TC −13; the float path returned −14 — M2 review). The float `decks_remaining` exists
only for the overlay and must never feed index decisions.

Known, intended divergences from the trainer's displayed values (trainer-notes §5): the trainer's
`DecksLeft` includes a `+1` and counts dealt-but-unrevealed holes; ours counts only *seen* cards —
the correct real-world Hi-Lo semantics (you can't count a card you haven't seen; never-revealed
holes stay uncounted, exactly as at a physical table). Consequence for evals: **RC is compared
exactly** (both sides count only revealed cards); TC/DecksLeft are compared with a documented
tolerance, normalized for the +1/hole-in-flight differences and the trainer's split-count quirk
via SPLIT events (finding 7).

## 6. Strategy engine (`strategy/`)

- `hands.py`: `HandView(total, is_soft, pair_rank, n_cards)` from `tuple[Rank, ...]`; legality
  predicates take `(hand_view, rules, ctx: TableContext)`:
  `can_double` = 2 cards ∧ DoubleRule satisfied; `can_split` = pair ∧ `ctx.num_hands` <
  `rules.max_hands` (aces: `max_ace_hands`); `can_surrender` = 2 cards ∧ `ctx.num_hands == 1`
  (surrender is illegal after any split) ∧ rules allow vs this upcard. (Finding 6 — a lone
  HandView cannot know split state; TableContext carries it.)
- `tables.py`: H17 base charts (hard/soft/pairs) as dicts, plus the 6-cell S17 diff — data
  transcribed from `docs/research/strategy-tables.md`. **Tests re-parse the research doc's CSV
  blocks independently and assert cell-for-cell parity** (the doc is the fixture source; the
  engine never parses the doc at runtime).
- `deviations.py`: I18 + Fab 4 entries `(play, hand_key, upcard, index, direction)` with
  `direction ∈ {at_or_above, below}` (below = strict, per research discrepancy #7 — boundary
  tests at index−1/index/index+1 mandatory). Rules-aware build: under H17, entries already basic
  (11vA double, 15vA surrender) are dropped; UNRESOLVED H17 indices use published S17 values
  (research §6).
- `engine.py` precedence: insurance flag (dealer A ∧ TC ≥ +3, independent of action) →
  surrender (deviation, then basic Rh/Rs) → split (deviation TT, then basic) → double/stand/hit
  (deviation, then basic). Deviations requiring an illegal action don't fire; hit/stand
  deviations apply to any card count. `Advice.fallback` lets the overlay show "D (else H)".
- **3+-card hands** (finding 10): vision cannot distinguish a doubled (locked) 3-card hand from a
  hit (live) one — bet chips aren't tracked. The engine still computes hit/stand advice (valid if
  live) and sets `Advice.caveat` so the overlay marks it "if hand still live". Documented
  limitation, advice-channel only. The same applies to locked split-ace hands:
  `Rules.hit_split_aces` is consumed by the RL env (P5), not by M1 advice legality.
- Bet suggestion (app-level, not engine): `bet_units = clamp(tc − 1, 1, 8)` — display only.

## 7. App shell wiring (`app/`)

Threading model (stack-notes §4/§5 constraints):

```
main thread:      DPI-awareness (ctypes, FIRST) -> load AppConfig -> tk root -> overlay
                  -> root.after(50ms) queue pump -> mainloop
pynput thread:    GlobalHotKeys{capture, reset, quit} -> job_queue.put(event)   [named combos only]
worker thread:    owns mss instance + detector + (RoundState, ShoeState)
                  job loop: grab -> detect -> reconcile -> apply count -> decide
                  -> ui_queue.put(OverlayModel)
```

- Single worker ⇒ frames serialized ⇒ no locking; state is replaced, never mutated.
- tkinter touched only by the main thread (queue + `after`), per stack-notes.
- **Shutdown lifecycle** (finding 11): quit hotkey → `job_queue.put(QUIT_SENTINEL)`; worker
  drains, closes mss, puts `QUIT` on `ui_queue`, exits; main-thread pump sees `QUIT` →
  `hotkeys.stop()` → `worker.join(timeout)` → `root.destroy()`. The sentinel unblocks the
  worker's blocking `get()`; both queues are unbounded so no `put` ever blocks.
- `AppConfig` (frozen): region, rules, conf threshold, hotkey combos, bet cap — persisted JSON
  (`config.json`, gitignored — contains machine-specific screen coords; committed
  `config.example.json` template).
- Region selector: one-time drag UI (tk fullscreen translucent), stores region; scale
  self-calibrates per frame (§4.3) so precision doesn't matter.
- Zero network calls anywhere in `src/` (Phase 7 greps for socket/http/urllib/requests imports).

## 8. RL chapter contracts (`rl/`)

- `env.py`: gymnasium-style `reset()/step()`; state = `(player_total, usable_ace,
  pair_rank_or_none, dealer_up, tc_bucket)` with `tc_bucket ∈ {≤−2, −1, 0, +1, +2, +3, ≥+4}`;
  actions filtered by the same `hands.py` legality predicates (with `TableContext`); dealing via
  `counting/shoe.py`'s seeded simulator at 6 decks / 75% penetration so TC states arise
  naturally; rules = the same frozen `Rules` (H17 reference config).
- `train.py`: tabular first-visit MC control, Q-table as numpy array keyed by state index;
  checkpoint to `models/rl_qtable.npz` + metadata JSON.
- `compare.py`: per-`tc_bucket` argmax-policy vs `strategy.engine` divergence heatmaps + EV via
  the M2 simulator. Gate: ≥97% agreement in the TC=0 bucket; every divergence vs published index
  reported honestly (match or mismatch) in `docs/RL-REPORT.md`.

## 9. Testing architecture

| Layer | Fixture source | Style |
|---|---|---|
| strategy | CSV/JSON blocks parsed from `docs/research/strategy-tables.md` by `tests/strategy/conftest.py` | Parametrized: every chart cell ×{H17,S17}; every deviation at index−1/index/index+1; ctx-legality cases (post-split surrender, split budget) |
| counting | seeded `random.Random` | Property loops: full shoe sums to RC 0; 10k shoes zero drift; TC flooring on negatives; decks_remaining clamp |
| tracker | `tests/fixtures/shoes/<name>/{frame_NNN.png, expected.json}` recorded at M5 **plus synthetic adversarial sequences targeting §13 findings 1–5**: dropped detection mid-round, captured surrender, identical consecutive short-settle deals, skipped settle, split relocations, suit flicker | Replay: exact RC at every frame; SUSPECT/ROUND_ENDED/PREV_ROUND_UNSETTLED classifications asserted |
| vision | held-out labeled frames from M3 + `models/metrics.json` | mAP50 ≥ 0.99, per-rank recall ≥ 0.995, latency <100 ms; fan-overlap NMS cases in the held-out split |
| app | `/verify` smoke path; M6 live 2-shoe test | Manual + scripted |

Coverage gates: strategy/counting ≥95%, tracker ≥85%, overall ≥80%.

## 10. Milestone → file map

| Milestone | Files |
|---|---|
| M1 | `types.py`, `strategy/*`, `tests/strategy/*` |
| M2 | `counting/*`, `tests/counting/*` |
| M3 | `vision/capture.py`, `vision/autolabel.py`, `data/REPORT.md` |
| M4 | `notebooks/train_yolo.ipynb`, `vision/detector.py` (ONNX impl), `models/*` |
| M5 | `tracker/*`, `vision/detector.py` (template impl), `tests/tracker/*`, `tests/fixtures/shoes/*` |
| M6 | `app/*` |
| P5 | `rl/*`, `tests/rl/*`, `docs/RL-REPORT.md` |

## 11. Key decisions log (delta over PRD's log)

| Decision | Choice | Rejected alternative | Why |
|---|---|---|---|
| Count derivation | Table-wide (rank,suit) multiset delta | Position-keyed card tracking | Immune to split relocation + skipped frames |
| Round boundaries | Positive-signal transition classifier (deal shape, hole transition, bbox continuity) | Multiset superset/cardinality inference | Review proved multiset-only boundaries silently corrupt the count (findings 1–3); "unknown transition" must fail safe (SUSPECT), not fail as "new round" |
| Shared types | Root-level `types.py` | Types scattered per package | strategy/counting/tracker must not import vision; one stdlib-only module breaks the cycle |
| `ShoeState.seen` | Full rank tuple | `(cards_seen: int, rc: int)` | Derived counts can't drift; reused by RL env + eval harness |
| tracker ↔ counting | Decoupled; worker applies revealed ranks to ShoeState | tracker owns the shoe | Both independently testable; count pipeline explicit in one place |
| Low-confidence / inconsistent frames | Reject whole frame (SUSPECT), warn, change nothing | Count high-conf subset | Partial counting corrupts silently; recapture is one keypress |
| Scale handling | Per-frame self-calibration from card width | Trust region selection / fixed 0.9 | Removes DPI + drag-precision failure modes |
| Detector classes | 53 (52 cards + "back") | 52 + heuristic hole detection | "back" is load-bearing for hole transitions and deal shape |
| Engine context | `TableContext` param into decide/legality | Single-hand signature | Post-split surrender illegality and split budgets are table-level facts (finding 6) |
| Eval TC comparison | RC exact; TC/DecksLeft normalized with tolerance | Match trainer's TC formula | Trainer's +1/hole-inclusive formula isn't real-world Hi-Lo; we keep correct semantics and normalize in the harness only (finding 8) |
| Property tests | Plain seeded `random.Random` loops | `hypothesis` dependency | Dependency-light; behaviors enumerable |

## 12. Risks accepted at this stage

- **Identical re-deal residual risk (finding 3):** a new deal that is pixel-identical to the
  previous accepted frame (same 3 exact cards after e.g. a player-blackjack settle) is
  indistinguishable from a hotkey double-tap by any vision-only system; it classifies as
  DUPLICATE and the new round's first frame goes uncounted until the next capture — RC error
  bounded by that frame's tags. Probability ≈ per-card collision³ (~10⁻⁶/occurrence);
  the DUPLICATE_FRAME warning tells the user to recapture after the next card. Accepted and
  documented rather than hidden.
- Template-matcher latency (~100–300 ms) exceeds the 100 ms detector budget — accepted for M5
  development only; the gate applies to the ONNX detector.
- The trainer split-count quirk (research §9.7) is an eval-harness concern: our count follows
  real Hi-Lo semantics; the harness normalizes using SPLIT events (§4.2) once the quirk is
  live-verified at M3.
- NMS-vs-fan-overlap coupling (finding 12) is mitigated structurally by the cardinality sanity
  check (§4.3), and empirically by fan-heavy held-out cases at M4.
- Multi-monitor negative coordinates: mss handles them; region selector stores absolute virtual
  coords (stack-notes §3).

## 13. Review record

Adversarial architect review (Opus) of the first draft produced 13 findings; all incorporated:

| # | Sev | Finding → resolution |
|---|---|---|
| 1 | CRIT | Dropped detection ⇒ false new round ⇒ survivor recount → transition classifier; SUSPECT fail-safe (§4.2) |
| 2 | CRIT | Captured surrender double-counts dealer upcard → ROUND_ENDED classification (§4.2) |
| 3 | CRIT | Identical short-settle re-deal missed; "settle ≥4 faces" claim false → positive deal signature + DUPLICATE handling + documented residual risk (§12) |
| 4 | HIGH | Count integrity depended on ungated suit accuracy → label-change gate at continuing positions (§4.2.0) |
| 5 | HIGH | Skipped settle silently under-counts → settledness tracking + PREV_ROUND_UNSETTLED + protocol amendment (§4.5, PRD updated) |
| 6 | HIGH | Single-hand decide ⇒ illegal advice (post-split surrender, split budget) → TableContext (§3, §6) |
| 7 | HIGH | events channel promised but absent; split-quirk normalization unimplementable → Event enum + SPLIT events (§3, §5) |
| 8 | MED | decks_remaining formula unpinned, diverges from trainer → §5 pins formulas + eval normalization policy |
| 9 | MED | Rules couldn't express double/split-limit configs → fields added (§3) |
| 10 | MED | Doubled (locked) hands get spurious advice → Advice.caveat, documented limitation (§6) |
| 11 | MED | Shutdown underspecified → sentinel + ordering (§7) |
| 12 | LOW | NMS vs 24px fan overlap → cardinality sanity check (§4.3) |
| 13 | LOW | last_hand/OverlayModel gaps → added to contracts (§3) |

## 14. Implementation amendments (Phase 4 review record)

Per-milestone review findings that changed contracts or pinned formulas:

| Milestone | Sev | Finding → resolution |
|---|---|---|
| M1 | HIGH | Splittable pair whose chart cell is non-split (7,7 vs T) skipped hard-total Fab4 surrender → surrender path now falls through to hard-total logic unless the pair cell is P/Ph(+DAS)/Rp |
| M1 | HIGH | Surrender fallback hardcoded HIT for Rs cells (dormant) → fallback derived from the chart code symmetrically |
| M1 | MED | `TableContext.hand_cards_count` dropped from §3 — `decide()` receives the cards and derives `HandView.n_cards`; `TableContext` carries `num_hands` only |
| M1 | MED | `Rules.hit_split_aces` documented as RL-env-only (§6); deviation lookups made O(1); `Advice` grafting via `dataclasses.replace` |
| M2 | CRIT | Float TC off-by-one at exact integer boundaries → §5 amended: exact integer floor-division `(RC·52)//remaining_cards`; float decks value is display-only; tests verify against `fractions.Fraction` ground truth |
| M2 | MED | `HILO_TAGS` wrapped in `MappingProxyType` (runtime-immutable); frozen+slots regression test added |

---
**Exit criterion (BUILD-GUIDE Phase 3):** user reviews and CONFIRMS this document. Phase 4 (TDD
implementation, starting M1) does not begin until then.
