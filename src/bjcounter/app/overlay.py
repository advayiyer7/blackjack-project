"""Always-on-top tkinter overlay, fed exclusively through a queue pumped on the main
thread via root.after (ARCHITECTURE §7 — tkinter is touched by the main thread only).

format_lines() is the pure, tested part; Overlay is the thin tk shell.
"""

from __future__ import annotations

import queue
import tkinter as tk
from collections.abc import Callable

from bjcounter.types import Action, Event, OverlayModel

PUMP_MS = 50
ACTION_NAMES = {
    Action.HIT: "HIT",
    Action.STAND: "STAND",
    Action.DOUBLE: "DOUBLE",
    Action.SPLIT: "SPLIT",
    Action.SURRENDER: "SURRENDER",
}
EVENT_NAMES = {
    Event.NEW_ROUND: "new round",
    Event.HOLE_REVEALED: "hole revealed",
    Event.SPLIT: "split",
    Event.ROUND_ENDED: "round ended",
    Event.DUPLICATE_FRAME: "duplicate frame",
    Event.PREV_ROUND_UNSETTLED: "PREV ROUND UNSETTLED",
}


def format_lines(model: OverlayModel) -> tuple[str, ...]:
    """OverlayModel -> display lines. Pure."""
    lines = [
        f"RC {model.running_count:+d}   TC {model.true_count:+d}   "
        f"decks {model.decks_remaining:.1f}",
        f"bet: {model.bet_units} unit{'s' if model.bet_units != 1 else ''}",
    ]
    if model.last_hand:
        lines.append("LAST HAND — shoe reshuffles next round; reset after it")
    for slot, advice in model.per_hand:
        if advice.action is None:
            text = advice.caveat or "no action"
        else:
            text = ACTION_NAMES[advice.action]
            if advice.fallback is not None:
                text += f" (else {ACTION_NAMES[advice.fallback]})"
        prefix = f"hand {slot + 1}: " if len(model.per_hand) > 1 else "play: "
        lines.append(prefix + text)
        if advice.is_deviation and advice.deviation:
            lines.append(f"  deviation — {advice.deviation}")
        if advice.caveat and advice.action is not None:
            lines.append(f"  ({advice.caveat})")
    if any(advice.insurance for _, advice in model.per_hand):
        lines.append("INSURANCE: take it (TC >= +3)")
    for event in model.events:
        lines.append(f"* {EVENT_NAMES[event]}")
    for warning in model.warnings:
        lines.append(f"! {warning}")
    return tuple(lines)


class Overlay:
    """Frameless topmost panel; drag anywhere to move; queue-fed."""

    def __init__(self, ui: queue.Queue, on_quit: Callable[[], None]) -> None:
        self._ui = ui
        self._on_quit = on_quit
        self._root = tk.Tk()
        self._root.overrideredirect(True)
        self._root.attributes("-topmost", True)
        self._root.geometry("+40+40")
        # External window destruction must still stop the hotkey listener; the
        # worker itself is covered by main()'s finally regardless of exit path.
        self._root.protocol("WM_DELETE_WINDOW", self._on_external_close)
        self._label = tk.Label(
            self._root,
            text="bjcounter — starting...",
            justify="left",
            anchor="nw",
            font=("Consolas", 11),
            bg="#101418",
            fg="#e8e8e8",
            padx=10,
            pady=8,
        )
        self._label.pack(fill="both", expand=True)
        self._label.bind("<Button-1>", self._drag_start)
        self._label.bind("<B1-Motion>", self._drag_move)
        self._drag_origin = (0, 0)

    def _on_external_close(self) -> None:
        self._on_quit()
        self._root.destroy()

    def _drag_start(self, event) -> None:
        self._drag_origin = (event.x, event.y)

    def _drag_move(self, event) -> None:
        x = self._root.winfo_x() + event.x - self._drag_origin[0]
        y = self._root.winfo_y() + event.y - self._drag_origin[1]
        self._root.geometry(f"+{x}+{y}")

    def _pump(self) -> None:
        try:
            while True:
                item = self._ui.get_nowait()
                if isinstance(item, str):  # QUIT sentinel from the worker
                    self._on_quit()
                    self._root.destroy()
                    return
                self._label.configure(text="\n".join(format_lines(item)))
        except queue.Empty:
            pass
        self._root.after(PUMP_MS, self._pump)

    def run(self) -> None:
        self._root.after(PUMP_MS, self._pump)
        self._root.mainloop()
