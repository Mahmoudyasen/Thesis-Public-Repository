"""Procedural blinding for WP4: nothing joins an NLST outcome to a model score unless the plan is committed and the
participant splits are the recorded ones (plan Sections 0, 5; NOTES 1)."""
from __future__ import annotations
import hashlib, subprocess
from pathlib import Path
from lungcad import config as C

REPO = Path(__file__).resolve().parents[2]
PLAN = "report/wp4_nlst_prespec.md"
SPLITS = C.NLST_PRE.parent / "meta" / "wp4_splits.csv"
SPLITS_SHA256 = "5db6a5b69b757de47523a5283a733b0aab479c3b65ba60d0e1f259f6497816cf"


def plan_commit() -> str | None:
    """Hash of the last commit touching the plan, or None if the plan is uncommitted or has uncommitted edits."""
    git = ["git", "-C", str(REPO)]
    h = subprocess.run(git + ["log", "-1", "--format=%h", "--", PLAN], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(git + ["status", "--porcelain", "--", PLAN], capture_output=True, text=True).stdout.strip()
    return h if h and not dirty else None


def require_unblinding_allowed(test: bool = False) -> dict:
    if test:
        return {"plan_commit": "TEST", "splits_sha256": "TEST"}
    h = plan_commit()
    if h is None:
        raise SystemExit(f"BLINDING: {PLAN} is not committed (or has uncommitted edits) - outcomes may not be joined to scores yet")
    sha = hashlib.sha256(SPLITS.read_bytes()).hexdigest()
    if sha != SPLITS_SHA256:
        raise SystemExit(f"BLINDING: {SPLITS} changed (sha256 {sha} != recorded {SPLITS_SHA256})")
    return {"plan_commit": h, "splits_sha256": sha}
