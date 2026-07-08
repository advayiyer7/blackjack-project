"""Global hotkeys via pynput. SECURITY RULE (BUILD-GUIDE §4.1): this module registers
the app's NAMED key combinations and nothing else — no key event logging, no
keystroke buffering, ever. pynput.GlobalHotKeys only invokes the callback when its
exact combination fires; individual keystrokes are never stored or forwarded.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

from pynput import keyboard


class Hotkeys:
    """Thin lifecycle wrapper around pynput.keyboard.GlobalHotKeys."""

    def __init__(self, bindings: Mapping[str, Callable[[], None]]) -> None:
        self._listener = keyboard.GlobalHotKeys(dict(bindings))

    def start(self) -> None:
        self._listener.start()

    def stop(self) -> None:
        self._listener.stop()
