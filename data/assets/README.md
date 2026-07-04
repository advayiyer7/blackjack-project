# Trainer sprite assets (dev-time only, gitignored)

`deck.png` and `table.png` are the Wizard of Odds trainer's own static art, fetched once for
the M3 auto-labeler templates and table-region location. They are NOT committed (third-party
art) and NOT used by the runtime app.

## Provenance (fetched 2026-07-03)

| File | Source URL | SHA-256 |
|---|---|---|
| deck.png (67×5452 sprite sheet: 52 faces @ 67×94 + back @ y=5358) | https://wizardofodds.com/wizfiles/play_ff/4/deck.png | 3105b27ea285620be4164acd28e9e579c245f46a89758512de5a3b26bc9fb6a5 |
| table.png (960×640 felt) | https://wizardofodds.com/wizfiles/play_ff/4/table.png | 4d529b5471ed12243757d0f5296253658c36fa3217dd8d8f6bbbcb43e015ee36 |

To re-fetch: download the two URLs above into this directory and verify the hashes.
