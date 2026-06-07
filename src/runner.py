"""
Full pipeline orchestration.

Runs all 6 stages in sequence:
  Stage 1  — async parallel classification
  Stage 2  — entity resolution (rapidfuzz + optional LLM)
  Stage 3  — evidence scoring
  Stage 4  — routing (partition into auto/review/monitor)
  Stage 5  — human review gate (CLI or skip in API mode)
  Stage 6  — streaming brief generation

Returns a PipelineResult with all intermediate state for inspection and logging.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from .brief import build_brief_metadata, stream_brief
from .classify import classify_all_sync
from .ingest import load_all
from .models import (
    Classification,
    ExecutiveBrief,
    ReviewDecision,
    RiskAssessment,
)
from .resolve import resolve_entities
from .review import (
    load_review_log,
    partition_assessments,
    run_cli_review,
    save_review_log,
)
from .score import score_items


@dataclass
class PipelineResult:
    classifications: list[Classification] = field(default_factory=list)
    assessments: list[RiskAssessment] = field(default_factory=list)
    auto_approved: list[RiskAssessment] = field(default_factory=list)
    needs_review: list[RiskAssessment] = field(default_factory=list)
    monitored: list[RiskAssessment] = field(default_factory=list)
    review_decisions: list[ReviewDecision] = field(default_factory=list)
    brief: ExecutiveBrief | None = None


def run_pipeline(
    cli_review: bool = True,
    reviewer_id: str = "reviewer",
    stream_to_stdout: bool = True,
) -> PipelineResult:
    """
    Execute the full pipeline.

    Args:
        cli_review:       If True, pause for human review of ambiguous items via CLI.
                          If False (API mode), human_review items are deferred for POST /review.
        reviewer_id:      Identifier stamped on review decisions.
        stream_to_stdout: If True, stream the final brief to stdout.
    """
    result = PipelineResult()

    print("Loading inputs...", flush=True)
    escalations, opportunities, projects = load_all()
    print(
        f"  {len(escalations)} escalations, {len(opportunities)} opportunities, "
        f"{len(projects)} projects loaded."
    )

    print("\nStage 1 — Classifying items (async parallel)...", flush=True)
    result.classifications = classify_all_sync(escalations, opportunities, projects)
    print(f"  {len(result.classifications)} items classified.")

    print("\nStage 2 — Resolving entities...", flush=True)
    entity_map = resolve_entities(escalations, opportunities, projects)
    resolved_count = len({e.entity_id for e in entity_map.values()})
    print(f"  {len(entity_map)} items resolved to {resolved_count} canonical entities.")

    print("\nStage 3 — Scoring evidence...", flush=True)
    result.assessments = score_items(
        escalations, opportunities, projects, result.classifications, entity_map
    )
    print(f"  {len(result.assessments)} assessments produced.")

    print("\nStage 4 — Routing...", flush=True)
    result.auto_approved, result.needs_review, result.monitored = partition_assessments(
        result.assessments
    )
    print(
        f"  Auto-approve: {len(result.auto_approved)} | "
        f"Human review: {len(result.needs_review)} | "
        f"Monitor: {len(result.monitored)}"
    )

    if cli_review:
        print("\nStage 5 — Human review gate...", flush=True)
        result.review_decisions = run_cli_review(result.needs_review, reviewer_id=reviewer_id)
        if result.review_decisions:
            save_review_log(result.review_decisions)
            print(f"  {len(result.review_decisions)} decision(s) recorded to review_log.json.")
    else:
        # API mode: load any previously submitted decisions
        result.review_decisions = load_review_log()

    # Build entity lookup maps for brief generation
    entity_names: dict[str, str] = {}
    entity_streams: dict[str, list[str]] = {}
    for entity in entity_map.values():
        entity_names[entity.entity_id] = entity.canonical_name
        entity_streams[entity.entity_id] = list(entity.source_streams)

    print("\nStage 6 — Generating executive brief...", flush=True)
    if stream_to_stdout:
        for chunk in stream_brief(
            result.auto_approved,
            result.needs_review,
            result.monitored,
            result.review_decisions,
            entity_names,
            entity_streams,
        ):
            # Strip SSE envelope for CLI output
            if chunk.startswith("data: ") and chunk != "data: [DONE]\n\n":
                text = json.loads(chunk[6:])
                print(text, end="", flush=True)
        print()  # newline after streaming completes

    result.brief = build_brief_metadata(
        result.auto_approved,
        result.needs_review,
        result.monitored,
        result.review_decisions,
        entity_names,
        entity_streams,
    )

    return result


if __name__ == "__main__":
    run_pipeline()
