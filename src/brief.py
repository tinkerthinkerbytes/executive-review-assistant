"""
Stage 6 — Brief Generation (streaming).

Takes auto-approved + human-approved P1/P2 items and generates
an executive briefing via OpenAI streaming.

Streaming output is yielded as SSE-compatible chunks.
The structured ExecutiveBrief is also assembled for downstream use.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Generator

from openai import OpenAI

from .models import BriefItem, ExecutiveBrief, ReviewDecision, RiskAssessment

MODEL = "gpt-4.1-nano"

BRIEF_SYSTEM_PROMPT = """You are preparing an executive briefing for a SaaS company's leadership team.
Your output answers one question: what needs management attention today?

Format:
## P1 — Immediate Attention Required
For each P1 item:
**[Account Name]** — [one-line headline]
Context: [2 sentences of relevant background]
Action: [specific recommended action]
Owner: [recommended owner]

## P2 — This Week
For each P2 item in the same format.

## P3 — Watch List
Brief bullets only.

Rules:
- Be direct. No filler. Executives read this in 90 seconds.
- Ground every claim in the evidence provided.
- Do not invent context not present in the input.
- If a human reviewer modified a recommendation, use their revised version."""


def _build_brief_context(
    approved: list[RiskAssessment],
    review_decisions: list[ReviewDecision],
    entity_names: dict[str, str],
    entity_streams: dict[str, list[str]],
) -> str:
    """Assemble the context block fed to the LLM."""
    # Index review decisions by item_id
    decisions_by_id = {d.item_id: d for d in review_decisions}

    sections = []
    for assessment in approved:
        name = entity_names.get(assessment.entity_id, assessment.entity_id)
        streams = entity_streams.get(assessment.entity_id, [])
        decision = decisions_by_id.get(assessment.item_id)

        action = assessment.recommended_action
        if decision and decision.decision == "modify" and decision.revised_recommendation:
            action = f"{decision.revised_recommendation} [REVIEWER OVERRIDE — original: {assessment.recommended_action}]"

        block = (
            f"Entity: {name} ({assessment.entity_id})\n"
            f"Risk tier: {assessment.risk_tier} | Evidence score: {assessment.evidence.score}/10\n"
            f"Streams: {', '.join(streams)}\n"
            f"Supporting: {'; '.join(assessment.evidence.supporting)}\n"
        )
        if assessment.evidence.contradicting:
            block += f"Contradicting: {'; '.join(assessment.evidence.contradicting)}\n"
        block += (
            f"Rationale: {assessment.evidence.rationale}\n"
            f"Action: {action}\n"
            f"Owner: {assessment.recommended_owner}\n"
        )
        sections.append(block)

    return "\n---\n".join(sections)


def stream_brief(
    auto_approved: list[RiskAssessment],
    human_approved: list[RiskAssessment],
    monitored: list[RiskAssessment],
    review_decisions: list[ReviewDecision],
    entity_names: dict[str, str],
    entity_streams: dict[str, list[str]],
) -> Generator[str, None, None]:
    """
    Yields SSE-compatible text chunks of the executive brief.
    Merges auto-approved and human-approved items, sorted by evidence score.
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Approved items = auto + human-approved (not skipped/modified to skip)
    decisions_by_id = {d.item_id: d for d in review_decisions}

    approved_items = list(auto_approved)
    for item in human_approved:
        d = decisions_by_id.get(item.item_id)
        if d is None or d.decision in ("approve", "modify"):
            approved_items.append(item)

    # Sort by evidence score descending
    approved_items.sort(key=lambda x: x.evidence.score, reverse=True)

    if not approved_items:
        yield "data: No items met the threshold for executive attention today.\n\n"
        return

    context = _build_brief_context(
        approved_items, review_decisions, entity_names, entity_streams
    )

    # Append watch list summary
    if monitored:
        watch = "\n\nP3/P4 Watch list (monitor only):\n" + "\n".join(
            f"- {entity_names.get(m.entity_id, m.entity_id)}: score {m.evidence.score}/10"
            for m in monitored
        )
        context += watch

    with client.chat.completions.stream(
        model=MODEL,
        messages=[
            {"role": "system", "content": BRIEF_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ],
        temperature=0.1,
    ) as stream:
        for text in stream.text_stream:
            yield f"data: {json.dumps(text)}\n\n"

    yield "data: [DONE]\n\n"


def build_brief_metadata(
    auto_approved: list[RiskAssessment],
    human_approved: list[RiskAssessment],
    monitored: list[RiskAssessment],
    review_decisions: list[ReviewDecision],
    entity_names: dict[str, str],
    entity_streams: dict[str, list[str]],
) -> ExecutiveBrief:
    """
    Build a structured ExecutiveBrief (non-streaming) for logging/storage.
    Brief text is generated separately via stream_brief.
    """
    decisions_by_id = {d.item_id: d for d in review_decisions}

    approved_items = list(auto_approved)
    for item in human_approved:
        d = decisions_by_id.get(item.item_id)
        if d is None or d.decision in ("approve", "modify"):
            approved_items.append(item)

    approved_items.sort(key=lambda x: x.evidence.score, reverse=True)

    p1_items, p2_items, p3_watch = [], [], []

    for assessment in approved_items:
        name = entity_names.get(assessment.entity_id, assessment.entity_id)
        streams = entity_streams.get(assessment.entity_id, [])
        decision = decisions_by_id.get(assessment.item_id)

        action = assessment.recommended_action
        if decision and decision.decision == "modify" and decision.revised_recommendation:
            action = decision.revised_recommendation

        brief_item = BriefItem(
            entity_id=assessment.entity_id,
            canonical_name=name,
            risk_tier=assessment.risk_tier,
            headline=assessment.evidence.rationale,
            context="; ".join(assessment.evidence.supporting),
            action=action,
            owner=assessment.recommended_owner,
            evidence_score=assessment.evidence.score,
            source_streams=streams,
        )

        if assessment.risk_tier == "P1":
            p1_items.append(brief_item)
        elif assessment.risk_tier == "P2":
            p2_items.append(brief_item)
        else:
            p3_watch.append(brief_item)

    # P3 watch includes monitored items too
    for assessment in monitored:
        name = entity_names.get(assessment.entity_id, assessment.entity_id)
        streams = entity_streams.get(assessment.entity_id, [])
        p3_watch.append(
            BriefItem(
                entity_id=assessment.entity_id,
                canonical_name=name,
                risk_tier=assessment.risk_tier,
                headline=assessment.evidence.rationale,
                context="; ".join(assessment.evidence.supporting),
                action=assessment.recommended_action,
                owner=assessment.recommended_owner,
                evidence_score=assessment.evidence.score,
                source_streams=streams,
            )
        )

    human_reviewed_count = len(
        [d for d in review_decisions if d.decision in ("approve", "modify")]
    )

    return ExecutiveBrief(
        generated_at=datetime.now(timezone.utc),
        p1_items=p1_items,
        p2_items=p2_items,
        p3_watch=p3_watch,
        items_reviewed=len(auto_approved) + len(human_approved) + len(monitored),
        items_auto_approved=len(auto_approved),
        items_human_reviewed=human_reviewed_count,
        items_monitored=len(monitored),
    )
