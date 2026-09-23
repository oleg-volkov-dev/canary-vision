"""Process-local canary routing and fixed-dataset promotion gates."""

import logging
from copy import deepcopy
from pathlib import Path
from secrets import randbelow
from threading import Event, Lock, Thread

from fastapi import HTTPException

from canary_vision.evaluation import evaluate
from canary_vision.model import Release


class Rollout:
    def __init__(self, classifier, manifest: Path, stage_seconds: float = 2):
        self._base = classifier
        self._stable = Release(classifier, "stable")
        self._candidate = None
        self._manifest = manifest
        self._lock = Lock()
        self._operation = Lock()
        self._percent = 0
        self._status = "idle"
        self._checks = []
        self._reason = None
        self._last_candidate = None
        self._previous_version = None
        self._automatic = False
        self._generation = 0
        self._stage_seconds = stage_seconds
        self._stop = Event()
        self._threads = []

    def close(self):
        self._stop.set()
        for thread in self._threads:
            thread.join()

    def snapshot(self):
        with self._lock:
            return deepcopy(
                {
                    "status": self._status,
                    "stable_version": self._stable.version,
                    "candidate_version": self._candidate.version if self._candidate else None,
                    "candidate_percent": self._percent,
                    "checks": self._checks,
                    "reason": self._reason,
                    "automatic": self._automatic,
                    "last_candidate_version": self._last_candidate,
                    "previous_version": self._previous_version,
                    "stable_weights_sha256": self._stable.weights_sha256,
                    "stable_weights_name": self._stable.weights_name,
                }
            )

    def predict(self, image):
        with self._lock:
            selected = (
                self._candidate
                if self._candidate and randbelow(100) < self._percent
                else self._stable
            )
        return selected.predict(image)

    def change(self, action: str, *, automatic: bool = False, candidate_factory=None):
        if not self._operation.acquire(blocking=False):
            raise HTTPException(409, "A rollout operation is already running.")
        try:
            if action in {"good", "bad", "custom"}:
                with self._lock:
                    if self._candidate:
                        raise HTTPException(409, "Finish or roll back the current candidate first.")
                candidate = (
                    candidate_factory() if action == "custom" else Release(self._base, action)
                )
                with self._lock:
                    if candidate.version == self._stable.version:
                        if not automatic:
                            raise HTTPException(409, "This release is already active.")
                        candidate.version += f"-r{self._generation + 1}"
                    self._generation += 1
                    self._candidate = candidate
                    self._last_candidate = candidate.version
                    self._previous_version = self._stable.version
                    self._percent = 10
                    self._status = "canary"
                    self._checks = []
                    self._reason = None
                    self._automatic = automatic
                if automatic:
                    self._threads = [t for t in self._threads if t.is_alive()]
                    thread = Thread(target=self._run, args=(self._generation,), daemon=True)
                    self._threads.append(thread)
                    thread.start()
            elif action == "rollback":
                with self._lock:
                    if not self._candidate:
                        raise HTTPException(409, "No candidate is active.")
                    self._rollback("Manual rollback. All traffic restored to the stable model.")
            elif action == "advance":
                with self._lock:
                    if self._automatic:
                        raise HTTPException(409, "Automatic checks are already running.")
                self._advance()
            else:
                raise HTTPException(422, "Unknown rollout action.")
            return self.snapshot()
        finally:
            self._operation.release()

    def _run(self, generation):
        while not self._stop.wait(self._stage_seconds):
            with self._operation:
                with self._lock:
                    if generation != self._generation or not self._automatic:
                        return
                try:
                    self._advance()
                except HTTPException:
                    return  # Evaluation errors already withdraw the candidate.

    def _advance(self):
        with self._lock:
            candidate = self._candidate
            if candidate is None:
                raise HTTPException(409, "No candidate is active.")
            self._status = "evaluating"
        try:
            report = evaluate(self._manifest, candidate)
        except Exception as exc:
            logging.getLogger(__name__).exception("Rollout evaluation failed")
            with self._lock:
                self._rollback("Evaluation could not complete; candidate withdrawn.")
            raise HTTPException(503, "Evaluation failed; candidate rolled back.") from exc
        with self._lock:
            self._checks.append({"candidate_percent": self._percent, **report})
            if not report["passed"]:
                self._rollback("Candidate failed the fixed-dataset accuracy gate.")
            elif self._percent == 100:
                self._stable = candidate
                self._candidate = None
                self._percent = 0
                self._status = "promoted"
                self._automatic = False
            else:
                self._percent = 50 if self._percent == 10 else 100
                self._status = "canary"

    def _rollback(self, reason):
        self._candidate = None
        self._percent = 0
        self._status = "rolled_back"
        self._automatic = False
        self._reason = reason
