"""Process-local canary routing and fixed-dataset promotion gates."""

import logging
from copy import deepcopy
from pathlib import Path
from secrets import randbelow
from threading import Lock

from fastapi import HTTPException

from canary_vision.evaluation import evaluate
from canary_vision.model import Release


class Rollout:
    def __init__(self, classifier, manifest: Path):
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

    def change(self, action: str):
        if not self._operation.acquire(blocking=False):
            raise HTTPException(409, "A rollout operation is already running.")
        try:
            if action in {"good", "bad"}:
                with self._lock:
                    if self._candidate:
                        raise HTTPException(409, "Finish or roll back the current candidate first.")
                    candidate = Release(self._base, action)
                    if candidate.version == self._stable.version:
                        raise HTTPException(409, "This release is already active.")
                    self._candidate = candidate
                    self._percent = 10
                    self._status = "canary"
                    self._checks = []
                    self._reason = None
            elif action == "rollback":
                with self._lock:
                    if not self._candidate:
                        raise HTTPException(409, "No candidate is active.")
                    self._rollback("Manual rollback.")
            else:
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
                    else:
                        self._percent = 50 if self._percent == 10 else 100
                        self._status = "canary"
            return self.snapshot()
        finally:
            self._operation.release()

    def _rollback(self, reason):
        self._candidate = None
        self._percent = 0
        self._status = "rolled_back"
        self._reason = reason
