# blackjack-counter

A real-time computer-vision card-counting advisor, built as an AI-engineering portfolio project.
On a global hotkey it screenshots the blackjack table region, detects every visible card with a
fine-tuned YOLO model, reconciles table state across frames, maintains a Hi-Lo running count and
true count, and displays the optimal action (hit / stand / double / split / surrender) from basic
strategy plus Illustrious 18 + Fab 4 count deviations in an always-on-top overlay.

Target surface: the [Wizard of Odds free blackjack trainer](https://wizardofodds.com/play/blackjack-v2/)
(6 decks, deep penetration, built-in count display used as ground truth for end-to-end evals).

**Status: work in progress.** Built phase-by-phase from [BUILD-GUIDE.md](BUILD-GUIDE.md).

## Ethics and framing (read this first)

This is a **training/practice tool for free-play simulators, not a tool for real-money play.**
Real-time assistance violates casino terms of service everywhere and constitutes a device offense
under statutes in some jurisdictions. Beyond legality, counting is mathematically dead on RNG
blackjack (the shoe is reshuffled every hand, so no count ever develops) and effectively neutered
on live-dealer tables (~50% penetration and 20–50 rounds/hour put expected value near ~$1/hour on
a $40k bankroll per published simulations). This project exists to practice counting skills against
a free trainer and to demonstrate an end-to-end CV + decision-engine + RL pipeline.

## Planned architecture

```
hotkey → capture (mss) → detect (YOLOv8n / ONNX CPU) → reconcile frames → Hi-Lo count → strategy engine → overlay (tkinter)
```

- `src/bjcounter/strategy/` — basic strategy tables + I18/Fab 4 deviations (pure, no I/O)
- `src/bjcounter/counting/` — Hi-Lo running/true count, shoe simulator (pure, no I/O)
- `src/bjcounter/vision/` — region capture, YOLO detector, template-match auto-labeler
- `src/bjcounter/tracker/` — frame reconciliation, round boundaries, split handling
- `src/bjcounter/rl/` — count-aware blackjack env + tabular RL agent (the portfolio chapter)
- `src/bjcounter/app/` — hotkeys, always-on-top overlay, wiring

## Privacy / security posture

- Screen capture is limited to the user-selected game-window region; raw screenshots are gitignored
  and never leave the machine.
- The runtime app makes **zero network calls** — inference is fully offline. The only network
  egress in the project is the Colab training notebook.
- The hotkey listener registers the app's specific hotkeys only — it is not, and will never be, a
  keylogger.

## Development

```
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
pytest
ruff check
```
