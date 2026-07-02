# Wizard of Odds Blackjack Trainer — Mechanics Notes

Research for building a CV card-detection pipeline against the WoO free blackjack
trainer. All findings below (except the "Needs manual verification" section) were
derived directly from downloaded page source, linked JS/CSS, and sprite assets —
not from guesswork. Source files were fetched on 2026-07-02 and saved to the
scratchpad for reference during this research pass (not committed to the repo).

## 1. Exact URL and which trainer to target

There are **two distinct games** under `/play/` on wizardofodds.com. They are
different codebases, not versions of the same file:

| URL | Asset path | What it is |
|---|---|---|
| `https://wizardofodds.com/play/blackjack/` | `/wizfiles/play_ff/2/blackjack.js?version=8.1` | The **original** ("first") trainer. Basic-strategy practice only. Options limited to Decks (1/2/6), DAS, Surrender, Soft17. **No penetration control, no card-counting system, no count display.** |
| `https://wizardofodds.com/play/blackjack-v2/` | `/wizfiles/play_ff/4/*` | The **card-counting trainer** ("second" trainer). Full rule customization, deck penetration control, 10 preset counting systems + custom values, running/true count display, "Analyze" combinatorial odds. |

The page body text on `/play/blackjack-v2/` states explicitly:

> "We are proud to present our second online blackjack trainer with the added
> ability to assist in learning the art of card counting, this is our first
> [linking to /play/blackjack/]. This is meant as an advanced tool, for those who
> have mastered basic strategy and are looking to perfect their card counting
> skills."

**Conclusion: `https://wizardofodds.com/play/blackjack-v2/` is the correct and
current target.** It was live and returned HTTP 200 with a full game page
(273KB HTML) at research time. No redirect. This doc covers v2 only from here on.

The game itself is embedded via a fixed `#game` div (960×640px canvas) with these
assets, all confirmed reachable (HTTP 200) directly under
`https://wizardofodds.com/wizfiles/play_ff/4/`:

```
blackjack.css?v=1
obj.js
misc.js
chipstack.js
rules.js?v=1
shoe.js
dealer.js
player.js
game.js
deck.png        (card sprite sheet, 67 x 5452 px)
table.png       (felt background, 960 x 640 px — exact size of #game)
arrow.png, bjpays.png, button.png, chip.png, chipshadow.png, rules.png, soft17.png
```

This is old-school "play_ff" (Flash-replacement) code: plain DOM `<div>` elements
positioned absolutely with inline pixel coordinates and CSS `background-position`
sprite offsets — **not** a `<canvas>`. No WebGL, no dynamically-generated card
art, no external image CDN dependency beyond wizardofodds.com itself.

## 2. Configurable settings (exact `<select>` values from page source)

All settings live in a "Rules" panel (`#rules`) gated behind an "adjust rules"
toggle, validated by `Rules.Validate()` in `rules.js`, and applied via
`Rules.CheckField`. Defaults noted below are what ships pre-selected on page
load (`selected` attribute in the HTML).

| Setting | Element id | Options | Default |
|---|---|---|---|
| Decks | `#decks` | 1, 2, 4, 5, 6, 8 | **8 decks** |
| Penetration | `#pen` | 50/55/60/65/70/75/80/85/90 % | **75%** |
| Dealer peek | `#peek` | peeks for blackjack / does not peek (ENHC) | peeks |
| Soft 17 | `#hitsoft17` | stands on all 17s (S17) / hits soft 17 (H17) | **hits soft 17** |
| Blackjack pays | `#bjpays` | 3:2, 7:5, 6:5, 1:1 | **3:2** |
| Split 2s–10s | `#splitX` | cannot / once / once-or-twice / up to 3 times | **up to 3 times** |
| Split Aces | `#splitA` | cannot / once / once-or-twice / up to 3 times | **once** |
| Hit split aces | `#hsa` | one card only / can receive multiple cards | **one card only** |
| Double down | `#dbl` | any two cards / hard 9,10,11 only / hard 10,11 only | **any two cards** |
| Double after split | `#das` | not after splitting / including after splitting | **including (DAS on)** |
| Surrender | `#surr` | not allowed / against any card / against any except Ace | **against any card** |
| Table min | `#tablemin` | $5–$10,000 | $5 |
| Table max | `#tablemax` | $300–$100,000 | $100,000 |
| Buy-in | `#buyin` | $100–$25,000+ | $5,000 |
| Counting system | `#ccsys` | Ace-Five, Hi-Lo, Hi-Opt I, Hi-Opt II, Insurance, KO, Omega II, OPP, Red 7, Zen, or custom per-rank values | **Hi-Lo** |
| Warn on mistakes | `#warnings` checkbox | on/off | **checked (on)** |
| Show count | `#showcount` checkbox | on/off | **checked (on)** |
| Turbo (animation speed) | `#turbo` checkbox | on/off | unchecked |

**For our target config (6 decks, 75–80% penetration):** both values are valid,
directly selectable options (`decks=6`, `pen=0.75` or `pen=0.80`). `rules.js`
client-side validation only blocks penetration/deck combos that are *too deep for
too few decks* (80%/85% require ≥2 decks, 90% requires ≥4 decks) — 6 decks clears
all of those, so no UI friction to reach 6-deck/75–80%.

Card counting values (`Rules.CountValue`, indices 0–10 = ranks 2,3,4,5,6,7,8,9,10,A,
+ custom "half" slot) and `Rules.CountIRC` (initial running count) are also
user-editable text fields, independently of the preset dropdown.

## 3. Card art — fixed digital sprites (deterministic, good for template matching)

Confirmed by downloading and inspecting the actual asset:

- `deck.png` is **67 × 5452 px**, a single-column vertical sprite sheet.
- Each card face occupies a fixed 67×94 px slot; CSS classes `.card0`…`.card51`
  set `background-position: 0 -94*Npx` for N = 0..51, and `.cardback` sits at
  `0 -5358px`.
- Cards are rendered as `<div class="card cardN">` elements (from
  `Element.prototype.CreateCard` in `obj.js`), **not** `<img>` tags and **not**
  `<canvas>` draws — pure CSS background-sprite divs, absolutely positioned in
  pixel coordinates within the 960×640 `#game` container.
- Card index `N` (0–51) maps onto rank via `Points[N]` (`misc.js`), a fixed
  52-entry array: each block of 13 consecutive indices repeats the rank sequence
  `2,3,4,5,6,7,8,9,10,10,10,10,11` (i.e., 2‑9, then four "10-value" ranks
  T/J/Q/K, then Ace=11). The four 13-index blocks correspond to the four suits,
  though nothing in the code labels which physical suit each block is — that
  would need a visual check against the sprite image itself (see manual
  verification list). For blackjack/count purposes rank alone is what matters,
  and rank is 100% deterministic from index.
- **The pixels are 100% fixed and identical every deal** — no per-session
  randomization of card art, no antialiasing differences, no dynamic text
  overlays baked into the card face. This is ideal for template matching /
  exact sprite-crop classification once the 52 card faces + back are extracted
  from `deck.png`.
- The table background (`table.png`, 960×640) exactly matches the `#game`
  container's native size — table felt is also a single static bitmap.
- Chips (`chip.png`), chip shadows (`chipshadow.png`), buttons (`button.png`),
  arrows, rules-toggle icon, and blackjack-payout/soft-17 signage
  (`bjpays.png`, `soft17.png`) are likewise static sprite sheets, not
  procedurally drawn.

Practical implication: a template-matching auto-labeler can pull the 53 fixed
card crops directly from `deck.png` (67×94 px each) as ground-truth templates,
rather than needing to harvest crops from live screenshots.

## 4. Layout regions (pixel coordinates, base 960×640 canvas)

The whole game renders inside `#game` at a **fixed 960×640px** logical size
(`background-image:url(table.png)`, `width:960px; height:640px`). Responsiveness
is done via a **CSS transform scale on the wrapper**, not responsive re-layout:

```css
@media (min-width: 1200px) { #gamemain.Blackjackv2 { transform: scale(0.9);  transform-origin: 0 50% 0; } }
@media (max-width: 1199px) { #gamemain.Blackjackv2 { transform: scale(0.74); transform-origin: 0 50% 0; } }
```

So there are exactly two known scale factors (0.9 and 0.74) depending on
viewport width, and all internal coordinates below are relative to the
unscaled 960×640 canvas — a screenshot pipeline should either force a viewport
≥1200px wide (for the 0.9 scale, larger/crisper cards) or account for the 0.74
scale, and can always recover exact card regions by dividing observed pixel
coordinates by the active scale factor.

Key regions (unscaled coordinates, from `player.js`/`dealer.js`/CSS):

- **Dealer hand**: `#dealerhand`, first card at `left:422px, top:10px`, each
  subsequent card offset `+24px` left, `0px` top (fans horizontally). Label
  (`#dealerhandlabel`) at `(422, 110)`.
- **Player hand(s)**: `#playerhand0..3` (4 possible hand slots for up to 3
  splits). Card offset per hand: `+24px` left, `-4px` top (slight upward fan).
  Starting coordinates depend on how many hands are in play (see split-layout
  table below). Each hand has its own label div `#playerhand{N}label`.
- **Count display** (`#countinfo`): a full-width black bar **above** the game
  table (inside `#gameoptions`, which sits above `#game`), not overlaid on the
  felt. It's plain text/HTML, not sprite-rendered.
- **Rules/config strip**: three single-line rule-summary spans (`#rules0`,
  `#rules1`, `#rules2`) rendered as small translucent white text in the
  **top-left corner of the table itself** (`left:5px; top:5,20,35px`),
  showing things like table min/max, "6D · 3:2 · PEEK · H17 · DA2", "SP3 · SPA1
  · DAS · NHSA · SURR", etc. Useful as an in-frame OCR-able ground truth of
  active rules if the pipeline ever needs to confirm rule config from a
  screenshot alone.
- **Bet/chip area**: chip selector at bottom-left (`chip0..3`, `y≈409-531`),
  bankroll stack at `(125,380)`, insurance chip stack at `(424,162)`.
- **Action buttons** (deal/hit/stand/double/split/insure/surrender/new-hand/
  repeat) are sprite-based, bottom-center/right of the table, each 90×90px.
- **Analysis output** (`#output`, only shown after clicking "Analyze"): a
  floating table at `(700, 230)`, width 232px, overlaid on top of the felt on
  the right side.
- **"LAST HAND" badge** (`#lasthand`): top-right corner, `left:850px, top:4px`.

## 5. Count display — confirmed ground truth for evals

Yes, confirmed in `shoe.js` (`Shoe.UpdateInfo`). The `#countinfo` bar renders:

```
Running Count: <b>{signed integer}</b>  |  Decks Left: {2 decimal places}  |  True Count: <b>{signed integer, rounded}</b>
```

- **Running count** is always shown (any counting system).
- **Decks Left** and **True Count** are only shown when `Rules.CountBalanced` is
  true — i.e., when the selected system is a "balanced" count (sums to zero
  across a full deck). The default preset, **Hi-Lo, is balanced**, so out of the
  box True Count and Decks Left both display. Unbalanced systems (e.g. KO,
  Insurance, Ace-Five as configured) would show Running Count only.
- Toggled on/off via the `#showcount` checkbox (checked/on by default);
  `Shoe.ToggleInfo()` just shows/hides the `#countinfo` div — the count value is
  still computed under the hood either way.
- True count formula: `TrueCount = RunningCount / DecksLeft`, where
  `DecksLeft = ((Decks*52) - CardsDealtSoFar + 1) / 52`, displayed rounded to
  nearest integer (`Math.round`); Decks Left itself is shown to 2 decimals.
- Count updates happen card-by-card as `Shoe.GetNextCard(update=true)` is
  called (i.e., live, not just at hand end), except the dealer's hole card
  specifically — it's dealt with `update:false` and only folded into the count
  via `Shoe.UpdateCard()` when it's actually revealed (peek-reveal or
  stand/hit resolution). This matters for building an eval harness: the
  trainer's own displayed running count only reflects revealed cards, matching
  real-table counting practice (you can't count a hole card you haven't seen).

This built-in display is a solid ground-truth oracle: an eval harness can OCR
or (better) hook/scrape `#countinfo`'s text content to compare against the
CV pipeline's independently-computed count.

## 6. Splits — layout and limits

- Up to **4 simultaneous hands** are supported in the DOM (`playerhand0..3`),
  matching `SplitX` max of "up to 3 times" (1 hand → split → 2 → split → 3 →
  split → 4 hands total, i.e. 3 splits max, 4 resulting hands max).
- Ace splitting is governed independently by `SplitA` (default: **once**, i.e.
  max 2 ace hands), and by default split aces receive **only one card each**
  (`HitSplitAces` default off) — matches standard casino rules.
- Split hands are **not simply "side by side" or "stacked" uniformly** — the
  trainer uses a bespoke fixed layout per hand-count, hard-coded in
  `player.js`'s `Split()` function (see the code's own coordinate comment
  table, reproduced here):

  | # hands | Hand 0 | Hand 1 | Hand 2 | Hand 3 |
  |---|---|---|---|---|
  | 1 | (443,416) | – | – | – |
  | 2 | (443,416) | (638,377) | – | – |
  | 3 | (196,358) | (443,416) | (638,377) | – |
  | 4 | (50,250) | (280,395) | (500,415) | (675,345) |

  (coordinates are the first-card top-left position for each hand; each hand's
  own cards then fan further by `+24px left, -4px top` per card as usual).
  This is a diagonal/staggered arc across the table, not a simple horizontal
  row — the auto-labeler / region-of-interest logic needs these exact
  per-hand-count layouts rather than assuming N evenly-spaced boxes.
- Each split triggers an animated card move (existing second card slides to
  the new hand position) plus a bankroll→wager chip animation for the new
  hand's matching bet. Splitting an Ace or pair still updates the running
  count (`Shoe.RunningCount += Rules.CountValue[10]` — cost of a phantom card
  discount used purely for insurance-count bookkeeping quirk in this codebase;
  worth independent verification if you rely on it, see note below).

## 7. Shuffle indication

- There is **no animated "shuffling" sequence and no discard-tray graphic**.
  The shoe re-initializes (`Shoe.Initialize()` → fresh full shoe + reshuffle)
  **instantly and silently** at the start of the next round once
  `Shoe.Next >= Shoe.CutCard` (`Shoe.BeginRound()` in `shoe.js`), where
  `CutCard = floor(Decks * 52 * Penetration)`.
- The **only pre-shuffle visual cue** is a **"LAST HAND" badge**
  (`#lasthand`, top-right of table, styled as a yellow rounded box) that
  appears as soon as the cut card has been reached mid-shoe
  (`Obj.LastHand.ShowIf(Shoe.Next >= Shoe.CutCard)` inside
  `Shoe.GetNextCard`). It's shown for the remainder of that hand, signaling
  "this is the last hand before the shoe reshuffles."
- After that hand resolves, the next `NewHand()`/`Repeat()` call runs
  `Shoe.BeginRound()`, which silently reshuffles if past the cut card — there
  is no on-screen message announcing the reshuffle itself, only the running
  count visibly resetting to the system's initial running count (`Rules.CountIRC`,
  default 0) on the following hand.
- Practical implication for a CV/eval pipeline: **detect shoe boundaries by
  watching for the "LAST HAND" badge appearing/disappearing and/or a
  discontinuous drop in running-count magnitude between hands**, not by
  looking for a shuffle animation (there isn't one).

## 8. Other useful implementation details

- Card-dealing/flip animation duration is 250ms normally, or 50ms (deal) /
  10ms (chip slides) with **Turbo mode** checked — purely a timing change, no
  visual/layout difference, so turbo mode doesn't affect template matching.
- "Warn on strategy errors" (`#warnings`, on by default) calls a server-side
  combinatorial solver (`GET /calculators-js/blackjack/calculate/`) via jQuery
  AJAX for every decision point, and blocks/warns before allowing a
  sub-optimal Hit/Stand/Double/Split/Surrender/Insurance choice. This means
  the page makes live network calls during play — relevant if the CV harness
  ever needs to run against a fully offline/mocked instance.
- No `<canvas>` anywhere in the game; everything is absolutely-positioned DOM
  with CSS sprites, which also means standard DOM/CSS scraping (not just
  pixel CV) is technically possible as a fallback/cross-check, though the
  project's stated goal is a vision pipeline.

## 9. Needs manual verification (M3)

These require actually loading and interacting with the live page in a real
browser (not just static source/asset inspection), and were **not** verified
in this pass:

1. **Actual on-screen pixel size of a dealt card** at a real browser viewport,
   after the 0.9/0.74 CSS transform is applied and combined with OS/browser
   DPI scaling — needed to calibrate the CV pipeline's expected card
   bounding-box size in real screenshots.
2. **Which physical suit corresponds to which 13-index block** in `deck.png`
   (blocks are index 0-12, 13-25, 26-38, 39-51) — doesn't matter for count
   value but would matter if the labeler also wants suit-level ground truth.
   Requires opening `deck.png` and visually inspecting/cropping.
3. **Exact rendered appearance/verification of the "LAST HAND" badge and count
   bar** in a live session (confirm no additional overlays, ads, or popups
   obscure these regions in practice; confirm no lazy-loaded/late-arriving
   iframes shift the table's vertical position on the page).
4. **Whether the page's surrounding chrome (nav bar, casino ads, footer) can
   push the game table's absolute position on the page** — the download shows
   the table is embedded mid-page within a large marketing page (casino
   reviews, software provider logos, etc.), not a standalone game view. A
   screenshot/automation pipeline will need to determine a stable way to
   locate/scroll to `#game` (e.g., via selector-based crop rather than fixed
   screen coordinates) — not confirmed how much vertical scroll offset varies
   run to run (ads above the fold could change table position).
5. **Live confirmation that the AJAX-based "warn on mistakes" feature doesn't
   introduce visible loading spinners/UI states** that could confuse a frame-
   by-frame CV pipeline (code shows it just delays button re-enablement via
   `setTimeout` polling, no visible spinner was found in CSS, but not
   confirmed visually).
6. **Cross-browser rendering consistency** (Chrome vs Firefox vs Safari) of
   the CSS background-position sprite technique — should be pixel-identical
   in theory, not verified in practice.
7. **Behavior of `Shoe.RunningCount += Rules.CountValue[10]` on every split**
   (seen in `player.js` `Split()`) — this adds "half the 10-count" value on
   each split, which looks like an unusual/possibly-quirky implementation
   detail (possibly a bug or an intentional insurance-side-count nuance in
   this codebase). Should be verified against the live count display when
   testing splits, since the eval harness will diff against this exact number.
8. **Whether closing/reopening the rules panel or changing a setting
   mid-shoe forces an immediate reshuffle** vs. waiting for `NewHand()` — code
   suggests rules changes only take effect through `Obj.Rules.Hide()` →
   returning to game state, but the exact reshuffle trigger point wasn't
   traced end-to-end interactively.

## Appendix: source files retrieved

Downloaded during this research pass (paths under this session's scratchpad,
not part of the repo):
`bjv2.html` (full page source, 273,831 bytes), `bjv1.html` (the older
`/play/blackjack/` page, 284,484 bytes, for comparison), `obj.js`, `misc.js`,
`chipstack.js`, `rules.js`, `shoe.js`, `dealer.js`, `player.js`, `game.js`,
`blackjack.css`, `deck.png` (67×5452, confirmed via PowerShell
`System.Drawing.Image`), `table.png` (960×640, confirmed same way).
