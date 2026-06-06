"""
Stage 1 — Classify: async parallel LLM classification of all input items.

Each item is classified simultaneously via asyncio.gather.
Returns key_signals, sentiment, and urgency_indicators per item.
"""

from __future__ import annotations

import asyncio
import os

from openai import AsyncOpenAI

from .models import Classification, Escalation, Opportunity, ProjectUpdate

MODEL = "gpt-4.1-nano"

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def _escalation_prompt(item: Escalation) -> str:
    return f"""Classify this customer escalation for executive review.

Account: {item.account_name}
Tier: {item.tier} | ARR: £{item.arr_gbp:,}
Issue: {item.issue_summary}
Days open: {item.days_open} | Escalations (30d): {item.escalations_last_30d}
Severity: {item.severity}

Extract key_signals (2-4 concise risk/urgency signals from the content),
sentiment (positive/neutral/negative/mixed toward the account relationship),
and urgency_indicators (explicit deadlines, time pressures, or escalation flags)."""


def _opportunity_prompt(item: Opportunity) -> str:
    return f"""Classify this sales opportunity for executive review.

Account: {item.account_name}
Tier: {item.tier} | ARR: £{item.arr_gbp:,}
Stage: {item.stage} | Deal value: £{item.deal_value_gbp:,}
Days stalled: {item.days_stalled} | Renewal date: {item.renewal_date}
Notes: {item.notes}

Extract key_signals (2-4 concise risk/urgency signals from the content),
sentiment (positive/neutral/negative/mixed toward deal outcome),
and urgency_indicators (explicit deadlines, time pressures, or risk flags)."""


def _project_prompt(item: ProjectUpdate) -> str:
    return f"""Classify this project status update for executive review.

Account: {item.account_name} | Project: {item.project_name}
Status: {item.status} | Phase: {item.phase}
Days overdue: {item.days_overdue} | Budget variance: {item.budget_variance_pct}%
Notes: {item.notes}

Extract key_signals (2-4 concise risk/urgency signals from the content),
sentiment (positive/neutral/negative/mixed toward delivery outcome),
and urgency_indicators (explicit deadlines, time pressures, or delivery risk flags)."""


async def _classify_item(
    item_id: str,
    item_type: str,
    prompt: str,
) -> Classification:
    client = _get_client()
    result = await client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an analyst preparing inputs for an executive briefing system. "
                    "Extract signals from operational data. Be concise and factual."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        response_format=Classification,
        temperature=0,
    )
    parsed = result.choices[0].message.parsed
    # Ensure item_id and item_type are set correctly regardless of what the model returns
    parsed.item_id = item_id
    parsed.item_type = item_type  # type: ignore[assignment]
    return parsed


async def classify_all(
    escalations: list[Escalation],
    opportunities: list[Opportunity],
    projects: list[ProjectUpdate],
) -> list[Classification]:
    """Classify all items in parallel via asyncio.gather."""
    tasks = []

    for item in escalations:
        tasks.append(
            _classify_item(item.id, "escalation", _escalation_prompt(item))
        )
    for item in opportunities:
        tasks.append(
            _classify_item(item.id, "opportunity", _opportunity_prompt(item))
        )
    for item in projects:
        tasks.append(
            _classify_item(item.id, "project", _project_prompt(item))
        )

    results = await asyncio.gather(*tasks)
    return list(results)


def classify_all_sync(
    escalations: list[Escalation],
    opportunities: list[Opportunity],
    projects: list[ProjectUpdate],
) -> list[Classification]:
    """Synchronous wrapper for use in CLI/runner context."""
    return asyncio.run(classify_all(escalations, opportunities, projects))
