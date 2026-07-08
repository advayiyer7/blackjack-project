# Model artifacts

Fine-tuned YOLO weights land here from `notebooks/train_yolo.ipynb` (`best.pt`,
`best.onnx`, fallback exports, `metrics_colab.json`); `scripts/eval_detector.py`
writes the authoritative local gate results to `metrics.json`.

## Checkpoint provenance (security rule §4)

| Artifact | Source | SHA-256 | Date |
|---|---|---|---|
| yolov8n.pt (pretrained base) | auto-downloaded by ultralytics 8.4.86 (github.com/ultralytics/assets releases) | f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36 | 2026-07-07 |
| best.pt (run 1: imgsz 1280, 50 epochs, torch 2.11.0+cu128, Colab T4) | trained in `notebooks/train_yolo.ipynb` | 41814d9cbe04930b1a62e0bba8621ffc145a85848691ffaa8a46c00768d46ee3 | 2026-07-07 |
| best.onnx / best_960.onnx (run 1 exports, opset 17) | exported from run-1 best.pt | see metrics_colab.json | 2026-07-07 |

Run-1 result on curated real test frames (see `metrics.json`): mAP50 1.000, per-rank
recall 1.000 at conf 0.80 — but 249 ms/frame CPU at the 1280 input. Run 2 retrains at
imgsz 640 (~55 ms measured locally) to close the latency gate; its weights replace
these files and this table gets a new row. Weights are committed once the M4 gates
pass in full.
