"""Capture worker: the single thread that owns capture, detection, and count state
(ARCHITECTURE §7). Frames are serialized through one thread, so there is no locking —
state objects are replaced, never mutated.

process_frame() is the pure heart (tested directly); Worker is the thin thread shell.
"""

from __future__ import annotations

import queue
import sys
import threading
import traceback
from enum import StrEnum
from pathlib import Path

from bjcounter.app.config import AppConfig
from bjcounter.counting.shoe import ShoeState
from bjcounter.strategy.engine import decide
from bjcounter.tracker.reconcile import step
from bjcounter.tracker.rounds import RoundState, fresh_shoe
from bjcounter.tracker.state import assemble_table
from bjcounter.types import Advice, OverlayModel, Rules, TableContext, TableState
from bjcounter.vision.detector import CardDetector


class Job(StrEnum):
    CAPTURE = "capture"
    RESET = "reset"
    QUIT = "quit"


CAPTURE, RESET, QUIT = Job.CAPTURE, Job.RESET, Job.QUIT
UiMessage = OverlayModel | Job


def process_frame(
    shoe: ShoeState,
    round_state: RoundState,
    table: TableState,
    rules: Rules,
    bet_cap: int,
) -> tuple[ShoeState, RoundState, OverlayModel]:
    """Reconcile one assembled frame into the count and advice. Pure."""
    result = step(round_state, table)
    if result.accepted:
        shoe = shoe.with_cards(card.rank for card in result.revealed)
        round_state = result.round_state

    tc = shoe.true_count
    per_hand: tuple[tuple[int, Advice], ...] = ()
    if result.accepted and table.dealer_cards and table.player_hands:
        upcard = table.dealer_cards[0].rank
        ctx = TableContext(num_hands=len(table.player_hands))
        per_hand = tuple(
            (hand.slot, decide(tuple(c.rank for c in hand.cards), upcard, tc, rules, ctx))
            for hand in table.player_hands
        )

    model = OverlayModel(
        running_count=shoe.running_count,
        true_count=tc,
        decks_remaining=shoe.decks_remaining,
        per_hand=per_hand,
        bet_units=max(1, min(tc - 1, bet_cap)),
        events=result.events,
        warnings=result.warnings,
        last_hand=table.last_hand,
    )
    return shoe, round_state, model


def status_model(shoe: ShoeState, warnings: tuple[str, ...] = ()) -> OverlayModel:
    """An OverlayModel that reports count state without a frame (startup/reset)."""
    return OverlayModel(
        running_count=shoe.running_count,
        true_count=shoe.true_count,
        decks_remaining=shoe.decks_remaining,
        per_hand=(),
        bet_units=1,
        events=(),
        warnings=warnings,
        last_hand=False,
    )


class Worker(threading.Thread):
    """Owns the mss grabber, the detector, and (RoundState, ShoeState).

    Created lazily inside run() because mss is thread-affine (vision/capture.py).
    On QUIT: drains, closes mss, forwards QUIT to the UI queue, exits (§7 shutdown).
    Non-daemon on purpose: the grabber must close before interpreter teardown —
    main() guarantees a QUIT job on every exit path.
    """

    def __init__(
        self,
        config: AppConfig,
        jobs: queue.Queue[Job],
        ui: queue.Queue[UiMessage],
    ) -> None:
        super().__init__(name="bjcounter-worker", daemon=False)
        self._config = config
        self._jobs = jobs
        self._ui = ui

    def _build_detector(self) -> CardDetector:
        if self._config.detector == "template":
            from bjcounter.vision.detector import TemplateMatchDetector

            return TemplateMatchDetector(Path(self._config.deck_png), self._config.scale)
        from bjcounter.vision.detector import OnnxYoloDetector

        return OnnxYoloDetector(
            Path(self._config.onnx_path), conf_threshold=self._config.conf_threshold
        )

    def run(self) -> None:
        from bjcounter.vision.capture import Grabber

        grabber = Grabber()
        try:
            detector = self._build_detector()
            shoe = ShoeState(seen=(), decks_total=self._config.rules.decks)
            round_state = fresh_shoe()
            frame_id = 0
            self._ui.put(status_model(shoe, ("ready — capture at the next deal",)))
            while True:
                job = self._jobs.get()
                if job == QUIT:
                    break
                try:
                    if job == RESET:
                        shoe = ShoeState(seen=(), decks_total=self._config.rules.decks)
                        round_state = fresh_shoe()
                        self._ui.put(status_model(shoe, ("count reset — fresh shoe",)))
                        continue
                    frame = grabber.grab(self._config.region)
                    table = assemble_table(
                        frame_id,
                        detector.detect(frame),
                        table_origin=self._config.table_origin,
                        conf_threshold=self._config.conf_threshold,
                    )
                    frame_id += 1
                    shoe, round_state, model = process_frame(
                        shoe, round_state, table, self._config.rules, self._config.bet_cap
                    )
                    self._ui.put(model)
                except Exception:  # noqa: BLE001 — worker must never die silently
                    print(traceback.format_exc(), file=sys.stderr)
                    self._ui.put(
                        status_model(
                            shoe, ("worker error — count unchanged; see console for details",)
                        )
                    )
        finally:
            grabber.close()
            self._ui.put(QUIT)
