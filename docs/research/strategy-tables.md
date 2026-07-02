# Blackjack Strategy Tables — Canonical Reference (for ~340 test fixtures)

Ruleset for all tables below unless noted otherwise: **6 decks, double after split (DAS) allowed,
late surrender (LS) allowed, double on any two cards, no re-splitting aces.** Both dealer
soft-17 conventions are covered (H17 = dealer hits soft 17; S17 = dealer stands on soft 17).
"No re-split aces" does not change any action code in these tables (A,A is always `P`
regardless of the resplit rule; the rule only affects what happens *after* the split).

Column header convention for every CSV block: `hand,2,3,4,5,6,7,8,9,T,A` (T = ten-value card,
A = dealer ace).

Action codes (as specified): `H` hit, `S` stand, `D` double else hit, `Ds` double else stand,
`P` split, `Ph` split if DAS offered else hit, `Rh` surrender else hit, `Rs` surrender else
stand, `Rp` surrender else split.

---

## 1. Basic Strategy — H17 (6D, DAS, LS, DA2, dealer hits soft 17)

Cross-checked cell-by-cell against two independent primary sources:
- Blackjack Apprenticeship's official "H17 Basic Strategy" chart (PDF, fetched directly and
  extracted verbatim — hard totals, soft totals, pairs, and late-surrender overlay tables).
- Wizard of Odds' "Basic Strategy in Text" bullet list (4-decks strategy page), which — though
  written as the S17 baseline — independently corroborates every hard-total, soft-total, and
  pair-splitting boundary that is *not* one of the confirmed H17/S17 exceptions (see §1.3).
- Blackjackinfo.com's basic-strategy-engine default description (6 decks, H17, DAS, LS, peek)
  independently confirmed the hard-10 row (double vs 2-9, hit vs 10/A).

### 1.1 Hard totals

```csv
hand,2,3,4,5,6,7,8,9,T,A
5,H,H,H,H,H,H,H,H,H,H
6,H,H,H,H,H,H,H,H,H,H
7,H,H,H,H,H,H,H,H,H,H
8,H,H,H,H,H,H,H,H,H,H
9,H,D,D,D,D,H,H,H,H,H
10,D,D,D,D,D,D,D,D,H,H
11,D,D,D,D,D,D,D,D,D,D
12,H,H,S,S,S,H,H,H,H,H
13,S,S,S,S,S,H,H,H,H,H
14,S,S,S,S,S,H,H,H,H,H
15,S,S,S,S,S,H,H,H,Rh,Rh
16,S,S,S,S,S,H,H,Rh,Rh,Rh
17+,S,S,S,S,S,S,S,S,S,Rs
```

Notes:
- Hard 5–7 are always hit; no reputable source doubles/splits these under 6D rules (below the
  first row that appears on any published chart, which is hard 8/9). This is uncontested.
- Row `17+` collapses hard 17, 18, 19, 20, 21 — all are always Stand except the surrender
  overlay vs dealer A (H17 only, see below).

### 1.2 Soft totals

Note: hand labels below contain a literal comma (e.g. `A,2`), so each is quoted to remain a
single CSV field — parse with a standard CSV reader (quote char `"`), not a naive `split(",")`.

```csv
hand,2,3,4,5,6,7,8,9,T,A
"A,2",H,H,H,D,D,H,H,H,H,H
"A,3",H,H,H,D,D,H,H,H,H,H
"A,4",H,H,D,D,D,H,H,H,H,H
"A,5",H,H,D,D,D,H,H,H,H,H
"A,6",H,D,D,D,D,H,H,H,H,H
"A,7",Ds,Ds,Ds,Ds,Ds,S,S,H,H,H
"A,8",S,S,S,S,Ds,S,S,S,S,S
"A,9",S,S,S,S,S,S,S,S,S,S
```

### 1.3 Pairs

Note: hand labels below contain a literal comma (e.g. `2,2`), so each is quoted to remain a
single CSV field — parse with a standard CSV reader (quote char `"`), not a naive `split(",")`.

```csv
hand,2,3,4,5,6,7,8,9,T,A
"2,2",Ph,Ph,P,P,P,P,H,H,H,H
"3,3",Ph,Ph,P,P,P,P,H,H,H,H
"4,4",H,H,H,Ph,Ph,H,H,H,H,H
"5,5",D,D,D,D,D,D,D,D,H,H
"6,6",Ph,P,P,P,P,H,H,H,H,H
"7,7",P,P,P,P,P,P,H,H,H,H
"8,8",P,P,P,P,P,P,P,P,P,Rp
"9,9",P,P,P,P,P,S,P,P,S,S
"T,T",S,S,S,S,S,S,S,S,S,S
"A,A",P,P,P,P,P,P,P,P,P,P
```

Derivation note: BJA's official pair chart only marks *whether* to split (Y / Y-if-DAS / N);
the action when **not** splitting is not printed in that table. Each non-split fallback above
was derived by mapping the pair to its equivalent hard total and reading that hard total's
verified action from §1.1 (e.g., 2,2 vs 8/9/T/A → hard 4 → always Hit; 5,5 → hard 10 →
Double 2-9 / Hit 10,A; 9,9 vs 7/T/A → hard 18 → always Stand; T,T → hard 20 → always Stand).
This mapping is standard/uncontested blackjack theory and is internally consistent with both
primary sources.

### 1.4 S17 differences (only cells that differ from the H17 chart above)

Derived from a direct cell-by-cell diff of Blackjack Apprenticeship's official H17 PDF vs. its
official S17 PDF (same publisher, same rule set — DAS/LS/6-8 deck — only the soft-17 dealer
rule toggled). Every other cell in hard totals, soft totals, and pairs is identical between the
two charts.

```csv
hand,upcard,h17_action,s17_action
11,A,D,H
15,A,Rh,H
17,A,Rs,S
"8,8",A,Rp,P
"A,7",2,Ds,S
"A,8",6,Ds,S
```

Confirmation of the two sanity-critical entries here:
- **11 vs A**: H17 = Double (always — it is basic strategy, no count needed); S17 = Hit.
  Independently corroborated by Wizard of Odds' S17 text: "Double hard 11 except against a
  dealer A."
- **A,8 vs 6**: H17 = Double-else-stand; S17 = plain Stand. Independently corroborated by the
  same WoO text, which lists soft-17/18 doubling as "vs. dealer 3-6" only for S17 (i.e., A,8's
  H17-only double vs 6 is the one WoO's S17 bullets omit).

---

## 2. Hi-Lo tag values

Verified against three independent sources (Wizard of Odds, PokerNews, Blackjack Apprenticeship
via general corroborating search) — full agreement, no discrepancy.

```csv
rank,tag
2,+1
3,+1
4,+1
5,+1
6,+1
7,0
8,0
9,0
T,-1
J,-1
Q,-1
K,-1
A,-1
```

---

## 3. True count conversion

**Formula:** `TC = running_count / decks_remaining`, where for a 6-deck shoe
`decks_remaining = (312 - cards_seen) / 52` (312 = 6 × 52).

**Rounding conventions** (per qfit.com's "True Count Calculation — The Whole Story", the
standard technical reference on this by Norm Wattenberger, author of CVData/CVCX — the software
used to generate most modern published index sets):

| Convention | Rule | Example (RC=+3, decks_remaining=2, raw TC=1.5) | Example (raw TC = -1.5) |
|---|---|---|---|
| Truncate (chop toward zero) | drop the fractional part, keep sign | 1 | -1 |
| Floor | always round down (toward -∞) | 1 | -2 |
| Round (nearest integer) | standard rounding | 2 | -1 (round-half-away or -2 round-half-even, implementation-dependent) |
| Half-deck resolution | resolve decks_remaining to the nearest 0.5 deck before dividing (a casual-play simplification, not used for generating index numbers) | — | — |

Truncation and flooring are identical for positive TC values; they diverge only for negative
TC (floor is more negative than truncate).

**Which convention do the Illustrious 18 assume?** Per a secondary citation of Don
Schlesinger's *Blackjack Attack* (3rd edition, p.190) surfaced via web search of a
blackjacktheforum.com discussion thread: "the methodology of choice for both reckoning the true
count and for subsequent use of indices is **flooring**." This is corroborated by qfit.com's
independent statement that flooring is "the most popular method used now" for index generation
and play, and performs statistically equivalently to plain rounding (both edge out truncation
slightly). **Conclusion: use flooring** (round toward -∞) when implementing TC-gated index
plays in this project — e.g., a raw TC of -0.3 floors to -1, not 0.

Caveat: the Schlesinger page-190 citation was obtained indirectly (via a search-engine summary
of a forum thread that could not be fetched directly — that specific forum URL returned HTTP
403). It is corroborated in substance by the independent qfit.com technical reference, so it is
not treated as unresolved, but it is flagged here as not independently confirmed by a directly
fetched primary quote.

---

## 4. Illustrious 18

Canonical Schlesinger ordering, verified for exact index number, hand, dealer upcard, and
direction across **three independent sources** (CountingEdge.com, Wizard of Odds
card-counting/high-low page, CasinoGuardian.co.uk) — all three agreed exactly on every one of
the 18 index values and on the ordering. This also matches the ordering given in the task
prompt exactly, confirming that ordering as correct.

```json
[
  { "play": "Insurance",      "hand": "any 2-card hand", "dealer_up": "A",  "index": 3,  "direction": "at_or_above", "h17_index_if_different": null },
  { "play": "Stand",          "hand": "16",   "dealer_up": "T", "index": 0,  "direction": "at_or_above", "h17_index_if_different": null },
  { "play": "Stand",          "hand": "15",   "dealer_up": "T", "index": 4,  "direction": "at_or_above", "h17_index_if_different": null, "note": "with LS available, basic strategy at TC<4 is Surrender (Rh); below TC 0 it is Hit. See discrepancies section for the 3-region resolution." },
  { "play": "Split",          "hand": "T,T",  "dealer_up": "5", "index": 5,  "direction": "at_or_above", "h17_index_if_different": null },
  { "play": "Split",          "hand": "T,T",  "dealer_up": "6", "index": 4,  "direction": "at_or_above", "h17_index_if_different": null },
  { "play": "Double",         "hand": "10",   "dealer_up": "T", "index": 4,  "direction": "at_or_above", "h17_index_if_different": null },
  { "play": "Stand",          "hand": "12",   "dealer_up": "3", "index": 2,  "direction": "at_or_above", "h17_index_if_different": null },
  { "play": "Stand",          "hand": "12",   "dealer_up": "2", "index": 3,  "direction": "at_or_above", "h17_index_if_different": null },
  { "play": "Double",         "hand": "11",   "dealer_up": "A", "index": 1,  "direction": "at_or_above", "h17_index_if_different": "N/A — 11vA is already basic strategy (always Double) under H17; the S17 index of +1 does not apply." },
  { "play": "Double",         "hand": "9",    "dealer_up": "2", "index": 1,  "direction": "at_or_above", "h17_index_if_different": null },
  { "play": "Double",         "hand": "10",   "dealer_up": "A", "index": 4,  "direction": "at_or_above", "h17_index_if_different": "UNRESOLVED — unverified forum claims of a 1-point shift (e.g. 4/3) could not be corroborated by two independent authoritative sources; not applied." },
  { "play": "Double",         "hand": "9",    "dealer_up": "7", "index": 3,  "direction": "at_or_above", "h17_index_if_different": null },
  { "play": "Stand",          "hand": "16",   "dealer_up": "9", "index": 5,  "direction": "at_or_above", "h17_index_if_different": null },
  { "play": "Hit",            "hand": "13",   "dealer_up": "2", "index": -1, "direction": "below", "h17_index_if_different": null },
  { "play": "Hit",            "hand": "12",   "dealer_up": "4", "index": 0,  "direction": "below", "h17_index_if_different": null },
  { "play": "Hit",            "hand": "12",   "dealer_up": "5", "index": -2, "direction": "below", "h17_index_if_different": null },
  { "play": "Hit",            "hand": "12",   "dealer_up": "6", "index": -1, "direction": "below", "h17_index_if_different": "UNRESOLVED — unverified forum claim of -3 under H17 could not be corroborated by two independent authoritative sources; not applied." },
  { "play": "Hit",            "hand": "13",   "dealer_up": "3", "index": -2, "direction": "below", "h17_index_if_different": null }
]
```

Direction semantics (Schlesinger/CVData convention — indices are "act at or above" numbers):
- `at_or_above` = take the listed `play` when `TC >= index`; otherwise play basic strategy.
- `below` = take the listed `play` (Hit) **strictly** when `TC < index`; at `TC >= index` play
  basic strategy (Stand). Equivalently: these five are stand-at-or-above-index entries whose
  basic action is already Stand, so the deviation fires only below the index. The boundary
  matters: 12 vs 4 has index 0 and the universally published phrasing is "hit 12v4 when the
  count is **negative**" — at exactly TC = 0 the play is Stand. Encoding these as
  "hit at TC <= index" would be off by one at the boundary in all five cases (see
  discrepancy #7).

---

## 5. Fab 4 surrenders

Verified across four sources (CountingEdge, Wizard of Odds, CasinoGuardian, and a general
web-search synthesis of Schlesinger's published numbers) with full agreement on the base
(commonly-published) index values.

```json
[
  { "play": "Surrender", "hand": "14", "dealer_up": "T", "index": 3, "direction": "at_or_above", "h17_index_if_different": null, "note": "Base action is Hit in both rule sets (neither BJA H17 nor S17 official chart lists 14 as a basic-strategy surrender)." },
  { "play": "Surrender", "hand": "15", "dealer_up": "T", "index": 0, "direction": "at_or_above", "h17_index_if_different": null, "note": "Same hand/upcard as an Illustrious 18 entry at index +4, but that entry governs a different action (Stand) at a different, higher threshold. Full picture for 15 vs T: Hit at TC<0, Surrender at 0<=TC<4, Stand at TC>=4. See discrepancies section." },
  { "play": "Surrender", "hand": "15", "dealer_up": "9", "index": 2, "direction": "at_or_above", "h17_index_if_different": null, "note": "Base action is Hit in both rule sets." },
  { "play": "Surrender", "hand": "15", "dealer_up": "A", "index": 1, "direction": "at_or_above", "h17_index_if_different": "UNRESOLVED for a specific number. Under H17, 15vA is ALREADY basic strategy Surrender at TC=0 (confirmed directly from BJA's official H17 chart), so the S17-oriented '+1 index to deviate from Hit to Surrender' does not apply as-is under H17. One source (CasinoGuardian) claimed an H17 value of -1 (implying an analogous low-TC Hit-crossover, structurally similar to the 15-vs-T three-region case) but this claim was internally inconsistent within that same source (its own table listed +1 with no rule-set qualifier) and could not be corroborated elsewhere. Treated as UNRESOLVED." }
]
```

---

## 6. Discrepancies & resolutions

Cross-checking surfaced the following items. Nothing below was filled from memory without at
least one primary-source chart or two independent secondary sources backing it.

1. **Task's own sanity-check premise was wrong and has been corrected.** The task prompt's
   sanity-check list states "8,8 vs T = P (Rp under some rules — verify for 6D LS)." Direct
   extraction of both Blackjack Apprenticeship official PDFs (H17 and S17) shows this is
   incorrect: **8,8 vs T (10) is plain `P` in both rule variants — never a surrender candidate.**
   The real Rp/P split is **8,8 vs A**: `Rp` under H17 (surrender if available, else split) and
   plain `P` under S17. This is independently corroborated by Wizard of Odds' S17 text, which
   explicitly carves out "surrender hard 16 **but not a pair of 8s**" with no mention of 10 vs
   8,8 surrendering either. Resolution: table §1.3/§1.4 use 8,8 vs **A**, not vs T.

2. **"15 vs T" appears in both the Illustrious 18 (index +4) and the Fab 4 (index 0) for what
   look like conflicting actions.** This is not a real conflict — the two entries govern
   different thresholds for the same hand: with late surrender available, the full picture for
   hard 15 vs dealer T is a three-region decision: **Hit** when TC < 0, **Surrender** when
   0 ≤ TC < 4, and **Stand** when TC ≥ 4 (the Illustrious 18's "16 vs 9" entry — index +5 — has
   the same structure: base action is Surrender, Illustrious 18 governs the high-TC Stand
   threshold). Resolution: documented explicitly in both JSON blocks rather than silently
   picking one.

3. **Illustrious 18 / H17 index differences beyond "11 vs A".** Multiple forum threads
   (blackjackinfo.com, blackjacktheforum.com) were searched for a documented H17-specific index
   table. Only **11 vs A** (becomes basic strategy — always double — under H17, so the S17
   index of +1 does not apply) was corroborated by more than one independent hit, and is also
   directly confirmed by comparing BJA's own H17 vs. S17 official charts (a primary-source,
   apples-to-apples comparison). Other candidate H17 shifts mentioned in passing on forums
   (T,T vs 6; 9 vs 2; 12 vs 6 → -3; 10 vs A → 3) could not be corroborated by two independent
   authoritative sources — one forum post explicitly attributed similar-looking numeric
   differences to **deck count** (4-deck vs 6-deck) rather than the H17/S17 rule, undermining
   the other claims. These are marked **UNRESOLVED** in §4's JSON rather than guessed.

4. **Fab 4 "15 vs A" H17 index.** CasinoGuardian's page was internally inconsistent — its own
   table listed the index as +1 with no rule-set qualifier, while separate prose on the same
   page claimed "+2 (S17) or -1 (H17)". Since (a) this single source contradicts itself and
   (b) no second independent source could confirm either alternate number, the H17-specific
   numeric index for 15 vs A is marked **UNRESOLVED**. The qualitative fact — that 15 vs A is
   already a basic-strategy surrender under H17 at TC=0, making the commonly-published "+1"
   (S17-oriented, Hit→Surrender threshold) inapplicable as published — is solidly confirmed via
   direct extraction of BJA's official H17 chart plus the absence of that surrender in BJA's
   S17 chart.

5. **blackjackincolor.com's Illustrious 18 extraction was discarded.** WebFetch's extraction of
   this page produced an internally inconsistent table (duplicate "12 vs 4" entries at two
   different ranks with different index values, and a mislabeled "12 vs 4: 3+" that should read
   "12 vs 2: +3" per every other source). This looks like a page-scraping/summarization
   artifact from an image-heavy page rather than a real second data point, so it was not counted
   as one of the corroborating sources; the three consistent sources (CountingEdge, Wizard of
   Odds, CasinoGuardian) were used instead.

6. **Wizard of Odds' interactive strategy calculator** (wizardofodds.com/games/blackjack/
   strategy/calculator/) could not be scraped — its rule-dependent chart is rendered
   client-side via JavaScript after the user picks options, which is invisible to a static HTML
   fetch. It was not used as a source; the static "4-decks" text-strategy page and the two
   Blackjack Apprenticeship PDFs were used instead, and agree with each other everywhere they
   overlap.

7. **Boundary convention for the five negative Illustrious 18 entries (corrected in review).**
   The first draft of §4 encoded 13v2, 12v4, 12v5, 12v6, 13v3 as "Hit at TC <= index"
   (`at_or_below`). That is off by one at the boundary under the canonical Schlesinger/CVData
   convention, in which every index is an "act at or above" number: stand at `TC >= index`, hit
   strictly below it. The corroborating check is 12 vs 4 (index 0): every standard reproduction
   phrases it as "hit 12v4 when the count is negative," i.e. at exactly TC = 0 the play is
   Stand, which only the strict-below encoding produces. §4 now uses `direction: "below"`
   (strict) for these five entries. Test fixtures must assert the boundary cell (TC exactly at
   index → basic strategy Stand).

No cells in the final hard-totals, soft-totals, or pairs tables (§1.1–1.3) were left
UNRESOLVED — full agreement was reached on all of them between the two primary sources.

---

## 7. Sources consulted (exact URLs, per table)

**Basic strategy (H17 chart, S17 differences, pairs derivation):**
- https://www.blackjackapprenticeship.com/wp-content/uploads/2024/09/H17-Basic-Strategy.pdf
- https://www.blackjackapprenticeship.com/wp-content/uploads/2024/09/S17-Basic-Strategy.pdf
- https://wizardofodds.com/games/blackjack/strategy/4-decks/
- https://www.blackjackinfo.com/blackjack-basic-strategy-engine/
- (attempted, not usable — client-side rendered) https://wizardofodds.com/games/blackjack/strategy/calculator/
- (attempted, not usable — image-only, HTTP 404/redirect issues) https://wizardofodds.com/games/blackjack/strategy/8-decks/ , https://easy.vegas/gambling/blackjack-basicstrategy.html

**Hi-Lo tag values:**
- https://wizardofodds.com/games/blackjack/card-counting/high-low/
- https://www.pokernews.com/casino/casino-terms/hi-lo-system.htm

**True count conversion & rounding convention:**
- https://gamblingcalc.com/gambling-guides/blackjack-card-counting/
- https://www.qfit.com/CalculatingTrueCounts.htm
- (indirect, via search synthesis; direct fetch returned HTTP 403) https://www.blackjacktheforum.com/showthread.php?26223-True-Count-Round-Floor-or-Truncate-CF-CV-Blackjack-and-CVData

**Illustrious 18 / Fab 4:**
- https://www.countingedge.com/blackjack-players/don-schlesinger/the-illustrious-18-card-counting-indices/
- https://wizardofodds.com/games/blackjack/card-counting/high-low/
- https://www.casinoguardian.co.uk/blackjack/blackjack-illustrious-18/
- (discarded as unreliable — see discrepancy #5) https://www.blackjackincolor.com/truecount3.htm
- (attempted, HTTP 500/403, not usable) https://www.qfit.com/illustrious18.htm , https://www.blackjacktheforum.com/showthread.php?44313-Illustrious-18-indexes
