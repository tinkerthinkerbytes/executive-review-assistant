"""
Stage 4 — Routing + Stage 5 — Human Review Gate.

Stage 4 partitions RiskAssessments into three buckets:
  auto_approve  — high-confidence P1/P2, score >= EVIDENCE_AUTO_THRESHOLD
  human_review  — P1/P2 with mixed signals, score 4-6
  monitor       — P3/P4

Stage 5 presents human_review items via CLI for approve / modify / skip.
Decisions are persisted to review_log.json and returned as ReviewDecision objects.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .models import ReviewDecision, RiskAssessment

REVIEW_LOG_PATH = Path(__file__).parent.parent / "review_log.json"


# ---------------------------------------------------------------------------
# Stage 4 — Routing
# ---------------------------------------------------------------------------

def partition_assessments(
    assessments: list[RiskAssessment],
) -> tuple[list[RiskAssessment], list[RiskAssessment], list[RiskAssessment]]:
    """
    Returns (auto_approved, needs_review, monitored).
    Deduplicates by entity_id: for each entity, keep the highest-scoring item.
    """
    # Deduplicate by entity: keep highest-scoring assessment per entity
    by_entity: dict[str, RiskAssessment] = {}
    for a in assessments:
        existing = by_entity.get(a.entity_id)
        if existing is None or a.evidence.score > existing.evidence.score:
            by_entity[a.entity_id] = a

    deduped = list(by_entity.values())

    auto_approved = [a for a in deduped if a.routing == "auto_approve"]
    needs_review = [a for a in deduped if a.routing == "human_review"]
    monitored = [a for a in deduped if a.routing == "monitor"]

    return auto_approved, needs_review, monitored


# ---------------------------------------------------------------------------
# Stage 5 — CLI Review Interface
# ---------------------------------------------------------------------------

def _present_item(assessment: RiskAssessment) -> None:
    e = assessment.evidence
    print("\n" + "=" * 60)
    print(f"  Item:          {assessment.item_id}")
    print(f"  Entity:        {assessment.entity_id}")
    print(f"  Risk tier:     {assessment.risk_tier}")
    print(f"  Evidence:      {e.score}/10")
    print(f"  Cross-stream:  {'Yes' if e.cross_stream else 'No'}")
    print(f"  Rationale:     {e.rationale}")
    print()
    print("  Supporting signals:")
    for s in e.supporting:
        print(f"    + {s}")
    if e.contradicting:
        print("  Contradicting signals:")
        for c in e.contradicting:
            print(f"    - {c}")
    print()
    print(f"  Recommended action:  {assessment.recommended_action}")
    print(f"  Recommended owner:   {assessment.recommended_owner}")
    print("=" * 60)


def run_cli_review(
    items: list[RiskAssessment],
    reviewer_id: str = "reviewer",
) -> list[ReviewDecision]:
    """
    Present each item for human review via CLI.
    Returns list of ReviewDecision objects.
    """
    decisions: list[ReviewDecision] = []

    if not items:
        print("\nNo items require human review.")
        return decisions

    print(f"\n{'='*60}")
    print(f"  HUMAN REVIEW GATE — {len(items)} item(s) require attention")
    print(f"{'='*60}")

    for assessment in items:
        _present_item(assessment)
        print("  Options: [a]pprove  [m]odify  [s]kip")

        while True:
            choice = input("  Decision: ").strip().lower()
            if choice in ("a", "approve"):
                decision = ReviewDecision(
                    item_id=assessment.item_id,
                    timestamp=datetime.now(timezone.utc),
                    decision="approve",
                    original_recommendation=assessment.recommended_action,
                    revised_recommendation=None,
                    reason=None,
                    reviewer_id=reviewer_id,
                )
                decisions.append(decision)
                break

            elif choice in ("m", "modify"):
                revised = input("  Revised recommendation: ").strip()
                reason = input("  Reason for override: ").strip()
                decision = ReviewDecision(
                    item_id=assessment.item_id,
                    timestamp=datetime.now(timezone.utc),
                    decision="modify",
                    original_recommendation=assessment.recommended_action,
                    revised_recommendation=revised,
                    reason=reason,
                    reviewer_id=reviewer_id,
                )
                decisions.append(decision)
                break

            elif choice in ("s", "skip"):
                decision = ReviewDecision(
                    item_id=assessment.item_id,
                    timestamp=datetime.now(timezone.utc),
                    decision="skip",
                    original_recommendation=assessment.recommended_action,
                    revised_recommendation=None,
                    reason=None,
                    reviewer_id=reviewer_id,
                )
                decisions.append(decision)
                break

            else:
                print("  Please enter 'a', 'm', or 's'.")

    return decisions


# ---------------------------------------------------------------------------
# Review log persistence
# ---------------------------------------------------------------------------

def load_review_log(path: Path | None = None) -> list[ReviewDecision]:
    p = path or REVIEW_LOG_PATH
    if not p.exists():
        return []
    raw = json.loads(p.read_text(encoding="utf-8"))
    return [ReviewDecision(**r) for r in raw]


def save_review_log(
    decisions: list[ReviewDecision],
    path: Path | None = None,
    append: bool = True,
) -> None:
    """Persist review decisions. Appends to existing log by default."""
    p = path or REVIEW_LOG_PATH
    existing: list[ReviewDecision] = load_review_log(p) if append else []

    # Avoid duplicate entries for the same item_id in the same run
    seen_ids = {d.item_id for d in existing}
    new = [d for d in decisions if d.item_id not in seen_ids]

    all_decisions = existing + new
    serialised = [
        {
            **d.model_dump(),
            "timestamp": d.timestamp.isoformat(),
        }
        for d in all_decisions
    ]
    p.write_text(json.dumps(serialised, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# API review submission (used by FastAPI /review/{item_id})
# ---------------------------------------------------------------------------

def submit_api_review(
    item_id: str,
    decision: str,
    original_recommendation: str,
    revised_recommendation: str | None,
    reason: str | None,
    reviewer_id: str,
    log_path: Path | None = None,
) -> ReviewDecision:
    """Create and persist a ReviewDecision from an API request."""
    d = ReviewDecision(
        item_id=item_id,
        timestamp=datetime.now(timezone.utc),
        decision=decision,  # type: ignore[arg-type]
        original_recommendation=original_recommendation,
        revised_recommendation=revised_recommendation,
        reason=reason,
        reviewer_id=reviewer_id,
    )
    save_review_log([d], path=log_path, append=True)
    return d
