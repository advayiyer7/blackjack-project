# Stack Notes — Phase 1 R2 Research

Verified against official docs/PyPI on **2026-07-02**. Where a search result looked
inconsistent across tools (dated info can be noisy for a future-dated environment),
the direct PyPI project page fetch was treated as the authoritative source.

Target stack recap: Windows 11, Python 3.12, hotkey-triggered `mss` region capture →
YOLO ONNX inference on CPU via `onnxruntime` → always-on-top `tkinter` overlay.
Training happens on Google Colab (free T4), export happens there too; only the
`.onnx` file ships to the local app.

---

## 1. Ultralytics YOLO — fine-tune, export, eval

**Version status:** pyproject pins `ultralytics==8.3.49` (released 2024-12-11).
Current PyPI stable is **8.4.86** (2026-07-02) — a large jump. Ultralytics ships
near-daily point releases, so treat any pin as a snapshot to re-verify, not a
long-lived pin. `torch>=1.8.0` is nominally accepted, but pin `torch` and
`ultralytics` together and re-test after any bump (see §torch note below and the
version table).

The library now also ships **YOLO11** (`yolo11n.pt`) and **YOLO26** (`yolo26n.pt`).
Both use the exact same `YOLO(...)`, `.train()`, `.export()`, `.val()` calls as
YOLOv8 — fully drop-in at the API level. Only the weights filename changes.

- **YOLO11** (2024-09-10): improved backbone/neck, e.g. YOLO11m matches/exceeds
  YOLOv8m mAP with ~22% fewer parameters.
- **YOLO26** (current flagship, marked stable/production in docs): removes DFL,
  and is **NMS-free by default** (`end2end=True`/one-to-one head is the default
  training mode). Ultralytics claims up to **43% faster CPU ONNX inference**
  for `yolo26n` vs `yolo11n` on an Intel Xeon CPU. For a CPU-only, real-time
  overlay app like this project, YOLO26 is worth benchmarking as the primary
  candidate — it removes the manual NMS step entirely (see §2).

### Fine-tune (Colab)

```python
from ultralytics import YOLO

# Start from a pretrained checkpoint. Swap the filename to try yolo11n.pt / yolo26n.pt.
model = YOLO("yolov8n.pt")

results = model.train(
    data="cards.yaml",   # dataset YAML, see below
    epochs=100,
    imgsz=640,
    batch=16,             # -1 = auto-batch based on free VRAM (useful on shared Colab GPUs)
    device=0,             # Colab T4 = GPU index 0
    pretrained=True,
    optimizer="auto",
)
```

### Dataset YAML (detection format)

```yaml
# cards.yaml
path: /content/cards_dataset   # dataset root
train: images/train            # relative to `path`
val: images/val
test: images/test              # optional

names:
  0: ace
  1: king
  2: queen
  # ... one entry per class, zero-indexed
```

Labels are YOLO-format `.txt` files (one per image, same basename), one line per
box: `class x_center y_center width height`, all normalized 0-1 relative to
image dimensions. Images with no objects need no label file.

### Export to ONNX

```python
model = YOLO("runs/detect/train/weights/best.pt")

model.export(
    format="onnx",
    imgsz=640,
    opset=17,        # pin explicitly — see onnxruntime compatibility note in §2
    dynamic=False,   # static shape is simpler/faster for a fixed capture region
    simplify=True,   # default True: runs onnxslim graph simplification
    nms=False,       # False = raw output, you run NMS yourself (see §2)
    # nms=True is also available on recent ultralytics: bakes IoU/conf thresholds
    # and NMS into the ONNX graph itself, at the cost of a fixed conf/iou at export time.
)
```

Key `export()` args: `format` (default `torchscript` — must set `"onnx"`),
`imgsz` (default 640), `opset` (default: highest supported by the installed
`onnx` package — **pin it explicitly**, do not rely on the default), `dynamic`
(False = fixed input shape, faster and simpler for this use case), `simplify`
(default True), `nms` (default False — adds NMS as graph ops when True), `half`
is deprecated in favor of `quantize=16`.

### Evaluate

```python
model = YOLO("best.pt")  # or best.onnx
metrics = model.val(data="cards.yaml", imgsz=640, conf=0.25, iou=0.7)

metrics.box.map      # mAP50-95
metrics.box.map50    # mAP50 — the number most worth tracking for card classes
metrics.box.map75
metrics.box.maps     # per-class mAP50-95, indexed by class id
# per-class precision/recall are available via metrics.box.p / metrics.box.r (arrays, one entry per class)
```

---

## 2. ONNX Runtime — CPU inference

**Version status:** pinned `onnxruntime==1.20.1` (Nov 2024). Current stable is
**1.27.0** (2026-06-15, requires Python ≥3.11 — compatible with our 3.12).
Recommend upgrading the pin, **and** explicitly pinning the ONNX `opset` at
export time (see §1) so a newer Colab-side `ultralytics`/`onnx` doesn't export
an opset the locally-pinned `onnxruntime` predates. Opset 17 is broadly
supported across `onnxruntime` 1.14+ through current and is a safe default;
bump it later only after confirming both sides of the pipeline in lockstep.

### Session creation

```python
import onnxruntime as ort

session = ort.InferenceSession(
    "best.onnx",
    providers=["CPUExecutionProvider"],  # explicit — don't let it silently try CUDA
)
input_name = session.get_inputs()[0].name    # typically "images"
output_name = session.get_outputs()[0].name  # typically "output0"
```

### Preprocessing (letterbox → NCHW float32)

```python
import cv2
import numpy as np

def preprocess(bgr_frame: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = bgr_frame.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(bgr_frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # letterbox pad to size x size, centered, gray (114) padding matches ultralytics default
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)      # BGR -> RGB
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0  # HWC -> CHW, normalize
    nchw = np.expand_dims(chw, axis=0)                  # add batch dim
    return nchw, scale, (top, left)
```

### Output shape and decode (NMS is NOT embedded by default)

For a standard (`nms=False`) Ultralytics detection ONNX export, the output
tensor shape is **`(1, 4 + num_classes, num_anchors)`** — e.g. `(1, 84, 8400)`
for 80-class COCO at imgsz=640 (`num_anchors` scales with `imgsz`). This is
**not** transposed the way most people expect: it's `[batch, box+cls, anchors]`,
so transpose to `(anchors, box+cls)` before working with it row-per-detection.

```python
outputs = session.run([output_name], {input_name: nchw})[0]  # (1, 4+nc, 8400)
preds = outputs[0].T  # (8400, 4+nc): [cx, cy, w, h, class0_score, class1_score, ...]

boxes_cxcywh = preds[:, :4]
class_scores = preds[:, 4:]
class_ids = class_scores.argmax(axis=1)
confidences = class_scores.max(axis=1)

conf_thresh = 0.25
keep = confidences > conf_thresh
boxes_cxcywh, class_ids, confidences = boxes_cxcywh[keep], class_ids[keep], confidences[keep]

# cxcywh -> xyxy for cv2.dnn.NMSBoxes / drawing
cx, cy, bw, bh = boxes_cxcywh.T
x1, y1 = cx - bw / 2, cy - bh / 2
boxes_xywh = np.stack([x1, y1, bw, bh], axis=1)  # cv2.dnn.NMSBoxes wants x,y,w,h

indices = cv2.dnn.NMSBoxes(
    boxes_xywh.tolist(), confidences.tolist(),
    score_threshold=conf_thresh, nms_threshold=0.45,
)
```

If instead you export with `nms=True` (or use a YOLO26 default end-to-end
export), the ONNX graph already returns filtered, NMS'd detections — no
manual NMS step needed, and the output shape/format changes to a fixed-size
`(1, max_det, 6)`-style `[x1, y1, x2, y2, conf, class_id]` tensor. This is the
main practical reason to evaluate YOLO26 for this project: it deletes the
`cv2.dnn.NMSBoxes` step and the decode is a straight slice.

Remember to un-letterbox the resulting boxes using the `scale`/`(top, left)`
returned by `preprocess()` before mapping back to the captured region's pixel
coordinates.

---

## 3. mss — region capture on Windows

**Version status:** pinned `mss==10.0.0`. Current stable is **10.2.0**
(2026-04-23, supports Python 3.10+). Recommend upgrading the pin.

### Region capture → numpy

```python
import mss
import numpy as np

region = {"left": 100, "top": 200, "width": 800, "height": 600}

with mss.mss() as sct:
    shot = sct.grab(region)
    img = np.asarray(shot)       # shape (h, w, 4), BGRA order
    bgr = img[:, :, :3]          # drop alpha -> BGR, already OpenCV-ready (no channel swap needed)
```

`shot` pixels are **BGRA** (blue, green, red, alpha) — slicing off the last
channel gives BGR directly, which is what OpenCV expects. No `cv2.cvtColor`
swap is needed for this step (only the ONNX preprocessing path needs a
BGR→RGB conversion, per §2).

### Multi-monitor

`sct.monitors[0]` is the union of all monitors (virtual screen); `sct.monitors[1:]`
are the individual physical monitors, each with its own `left`/`top` origin
(which can be negative for monitors positioned left of/above the primary).
Build the capture region dict from real coordinates, not assumed (0,0) origin,
if the target window can be on a secondary monitor.

### Windows DPI-scaling gotcha (critical for this project)

Windows 11 displays are very commonly scaled (125%, 150%, 175%...), not 100%.
Two separate risks:

1. **mss itself is coordinate-correct** as long as it can set
   `PROCESS_PER_MONITOR_DPI_AWARE` — it calls `SetProcessDpiAwareness(2)`
   internally on `mss.mss()` construction. `grab()` then returns real physical
   pixels for the region you asked for.
2. **But `SetProcessDpiAwareness` can only succeed once per process.** If
   *anything* else in the process (another library, or your own code) sets a
   different DPI-awareness mode first — or if you create the `tkinter` root
   window first and it implicitly ends up DPI-unaware — mss's coordinate
   system and tkinter's window/mouse coordinate system will disagree, and the
   overlay will be misaligned/scaled relative to the captured region,
   especially on multi-monitor setups with mixed scaling factors.

**Fix: set DPI awareness yourself, once, at the very top of `main()`, before
creating the `mss` context or the `Tk()` root:**

```python
import ctypes

def enable_dpi_awareness() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()   # Windows 7/8 fallback
        except (AttributeError, OSError):
            pass  # best-effort; degrade gracefully rather than crash

# call this first thing in your entrypoint, before mss.mss() and before tk.Tk()
enable_dpi_awareness()
```

Doing this before both mss and tkinter initialize guarantees both subsystems
agree on "physical pixels," so the overlay draws boxes at coordinates that
line up with the captured region regardless of Windows display scaling.

---

## 4. pynput — global hotkeys

**Version status:** pinned `pynput==1.7.7`. Current stable is **1.8.2**
(2026-05-12; recent changelog entries include Windows scroll-event fixes).
Recommend upgrading the pin. `GlobalHotKeys` API is unchanged across these
versions.

```python
from pynput import keyboard

def on_capture():
    ...  # enqueue a capture request; do the real work off this callback thread

def on_toggle_overlay():
    ...

hotkeys = keyboard.GlobalHotKeys({
    "<ctrl>+<alt>+c": on_capture,
    "<ctrl>+<alt>+h": on_toggle_overlay,
})
hotkeys.start()   # non-blocking: runs its own listener thread

try:
    ...  # main program / tkinter mainloop runs here
finally:
    hotkeys.stop()
    hotkeys.join()  # wait for the listener thread to fully exit
```

`GlobalHotKeys` is itself a `Thread` subclass; `.start()` returns immediately
and the hotkey matching happens on a background OS hook thread. Callbacks fire
on that background thread — never touch `tkinter` widgets directly from them
(see §5, use the queue+`after()` pattern). A stopped `GlobalHotKeys` instance
cannot be restarted; construct a new one if hotkeys need to change at runtime.

**Hard security/scope constraint for this project:** `GlobalHotKeys` (or any
lower-level `keyboard.Listener`) must only ever be wired to the small, fixed
set of registered combos this app defines (e.g. capture, toggle-overlay,
quit). Do **not** attach a generic `on_press`/`on_release` listener that logs,
buffers, or forwards arbitrary keystrokes — that would turn a hotkey trigger
into a keylogger. `GlobalHotKeys`'s dict-of-combos API naturally enforces
this (it only ever calls back on exact combo matches), so prefer it over the
raw `Listener` API specifically because it can't accidentally observe
unrelated keys.

---

## 5. tkinter — always-on-top overlay

```python
import tkinter as tk

root = tk.Tk()
root.overrideredirect(True)          # frameless: no titlebar/borders
root.attributes("-topmost", True)    # always-on-top
root.geometry(f"{w}x{h}+{x}+{y}")    # size + position over the captured region

# Transparency: pick ONE approach.
# (a) whole-window opacity (simple, affects everything drawn):
root.attributes("-alpha", 0.85)

# (b) colorkey transparency (Windows-only): any pixel drawn in this exact
#     color becomes fully see-through, rest of the window stays opaque:
TRANSPARENT_KEY = "#010203"  # pick a color nothing else in the UI will use
root.config(bg=TRANSPARENT_KEY)
root.attributes("-transparentcolor", TRANSPARENT_KEY)
```

### Click-through (optional, Windows-only, ctypes — no pywin32 dependency needed)

Plain Tk transparency (`-transparentcolor`/`-alpha`) still intercepts mouse
clicks — the window is *visually* transparent but not click-through. To let
clicks pass to the game window underneath, apply the Win32 extended window
styles after the window exists:

```python
import ctypes

GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020

def make_click_through(root: tk.Tk) -> None:
    hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
    styles = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
    ctypes.windll.user32.SetWindowLongW(
        hwnd, GWL_EXSTYLE, styles | WS_EX_LAYERED | WS_EX_TRANSPARENT
    )
```

Call this after `root.update()`/once the window is mapped. Note it makes the
*entire* window click-through, including any overlay controls you draw — only
use it on a pure display-only overlay, or toggle the style off while the user
needs to interact with an overlay button.

### Threading rule (must follow — tkinter is not thread-safe)

The hotkey listener (§4) and the capture/inference work both run on
non-main threads. `tkinter` widgets may only be touched from the thread
running `root.mainloop()`. Route all cross-thread updates through a
`queue.Queue`, drained by a `root.after(...)`-scheduled poller on the main
thread:

```python
import queue

update_queue: queue.Queue = queue.Queue()

def poll_queue():
    try:
        while True:
            job = update_queue.get_nowait()
            job(root)  # e.g. a closure that updates a label/canvas
    except queue.Empty:
        pass
    finally:
        root.after(50, poll_queue)  # ~20 Hz poll, adjust to taste

root.after(50, poll_queue)
root.mainloop()
```

Worker threads (hotkey callback, capture+inference loop) only ever call
`update_queue.put(...)` — never `root.<widget>.config(...)` directly, and
never call `mainloop`/create widgets off the main thread.

---

## Version pins to confirm

| Package | Our pin | Current stable found (2026-07-02) | Recommendation |
|---|---|---|---|
| `ultralytics` | `8.3.49` | `8.4.86` | Upgrade. Re-pin to a specific current version (don't float); re-run train/export/val smoke tests after bumping since Ultralytics ships near-daily releases. Evaluate `yolo26n.pt` as an alternative to `yolov8n.pt` — same API, claimed faster CPU ONNX inference, and NMS-free export removes the manual NMS step in §2. |
| `torch` (train extra) | `2.5.1` | ~`2.9`–`2.12` (search results were inconsistent; PyPI page fetch showed `2.12.1`, 2026-06-17) | Low priority to hard-pin: Colab preinstalls a torch build matched to its runtime's CUDA driver, so a strict local pin mostly matters for reproducibility/provenance, not for the Colab training run itself. If pinning, pin whatever Colab's current runtime ships rather than forcing an arbitrary version via pip (forcing a mismatched torch/CUDA pair on Colab is a common breakage source). |
| `onnxruntime` | `1.20.1` | `1.27.0` (2026-06-15, requires Python ≥3.11) | Upgrade. More importantly: pin the ONNX **opset** explicitly in `model.export(..., opset=17)` regardless of this package's version, so a newer Colab-side export doesn't emit an opset this pinned runtime predates. |
| `mss` | `10.0.0` | `10.2.0` (2026-04-23, Python 3.10+) | Upgrade. No API changes affecting the region-capture usage above; still relevant to re-test the DPI-awareness ordering (§3) after upgrading. |
| `pynput` | `1.7.7` | `1.8.2` (2026-05-12) | Upgrade. `GlobalHotKeys` API unchanged; changelog includes Windows-specific scroll-event fixes (not directly relevant to keyboard hotkeys, but confirms active Windows support). |

Note: `numpy==2.1.3`, `opencv-python==4.10.0.84`, and `gymnasium==1.0.0` are
also pinned in `pyproject.toml` but were out of scope for this research pass
(not among the five areas assigned); flag them for a follow-up pin check if
the ONNX/mss upgrades above end up requiring newer NumPy ABI compatibility.
