# Dataset progress report

Generated 2026-07-06 by scripts/dataset_report.py.

## Training pool (synthetic, data/synthetic/)

- Frames: **401** (regenerable — see synth_meta.json for the seed)
- Card instances: 4378
- Classes below the 40-instance train gate: **0** of 53

## Val/test pool (real captures, data/raw/)

- Sessions analyzed: 2
- Valid frames (cumulative): **123** / target ~1500
- Card instances total: 351
- Classes below the 40-instance gate: **53** of 53
- Player-hand layout coverage (frames): 1 hand=80, 2 hands=1, 3 hands=0, 4 hands=0

## Per-session (real captures)

| Session | Scale | Frames | Valid | Skipped | Mean score | Min score |
|---|---|---|---|---|---|---|
| session_20260703_172916 | 1.12 | 45 | 27 | 18 | 0.969 | 0.942 |
| session_20260705_172946 | 1.25 | 103 | 96 | 7 | 0.969 | 0.901 |

## Instances per class (real / synthetic)

| Class | Real | Synth | | Class | Real | Synth |
|---|---|---|---|---|---|---|
| 2c | 2 | 80 | | 3h | 0 | 80 |
| 3c | 7 | 80 | | 4h | 0 | 80 |
| 4c | 4 | 80 | | 5h | 8 | 80 |
| 5c | 7 | 80 | | 6h | 9 | 81 |
| 6c | 0 | 80 | | 7h | 6 | 80 |
| 7c | 8 | 81 | | 8h | 8 | 80 |
| 8c | 7 | 81 | | 9h | 2 | 80 |
| 9c | 14 | 81 | | Th | 17 | 80 |
| Tc | 5 | 81 | | Jh | 13 | 80 |
| Jc | 9 | 80 | | Qh | 5 | 80 |
| Qc | 10 | 80 | | Kh | 8 | 80 |
| Kc | 0 | 80 | | Ah | 18 | 80 |
| Ac | 4 | 80 | | 2s | 5 | 81 |
| 2d | 5 | 80 | | 3s | 11 | 80 |
| 3d | 10 | 80 | | 4s | 10 | 80 |
| 4d | 0 | 80 | | 5s | 7 | 80 |
| 5d | 7 | 80 | | 6s | 13 | 80 |
| 6d | 0 | 80 | | 7s | 9 | 80 |
| 7d | 13 | 80 | | 8s | 0 | 80 |
| 8d | 0 | 80 | | 9s | 1 | 80 |
| 9d | 0 | 80 | | Ts | 7 | 80 |
| Td | 3 | 80 | | Js | 9 | 80 |
| Jd | 8 | 80 | | Qs | 3 | 80 |
| Qd | 0 | 80 | | Ks | 9 | 80 |
| Kd | 9 | 80 | | As | 1 | 80 |
| Ad | 4 | 80 | | back | 36 | 211 |
| 2h | 0 | 81 | | | | |

## Real captures below target

2c (2), 2d (5), 2h (0), 2s (5), 3c (7), 3d (10), 3h (0), 3s (11), 4c (4), 4d (0), 4h (0), 4s (10), 5c (7), 5d (7), 5h (8), 5s (7), 6c (0), 6d (0), 6h (9), 6s (13), 7c (8), 7d (13), 7h (6), 7s (9), 8c (7), 8d (0), 8h (8), 8s (0), 9c (14), 9d (0), 9h (2), 9s (1), Ac (4), Ad (4), Ah (18), As (1), Jc (9), Jd (8), Jh (13), Js (9), Kc (0), Kd (9), Kh (8), Ks (9), Qc (10), Qd (0), Qh (5), Qs (3), Tc (5), Td (3), Th (17), Ts (7), back (36)
