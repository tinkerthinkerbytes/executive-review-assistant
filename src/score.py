"""
Stage 3 — Evidence Scoring + Cross-Stream Enrichment.

Scoring is rules-based and deterministic (0-10).
LLM writes the rationale sentence only — does not produce the score.

Scoring rules:
  Base: 0
  +1 per supporting signal (up to 4)
  -1 per contradicting signal (up to -2)
  +2 if entity appears in 2+ streams (cross-stream corroboration)
  +1 if Enterprise tier
  +1 if ARR > £100k
  +1 if days_open > 14 (escalations) or days_overdue > 0 (projects)
  +1 if escalations_last_30d >= 3 (escalations)
  +1 if stage == "Closed Lost" (opportunities — revenue already lost)
  +1 if project status == "Red"

Risk tier assignment:
  score >= 8 → P1
  score 6-7  → P2
  score 4-5  → P3
  score < 4  → P4

Routing:
  (P1 or P2) and score >= EVIDENCE_AUTO_THRESHOLD → auto_approve
  (P1 or P2) and score 4-6                         → human_review
  P3 or P4                                          → monitor
"""

from __future__ import annotations

import os

from openai import OpenAI

from .models import (
    Classification,
    EvidenceStrength,
    Escalation,
    Opportunity,
    ProjectUpdate,
    ResolvedEntity,
    RiskAssessment,
)

MODEL = "gpt-4.1-nano"
EVIDENCE_AUTO_THRESHOLD = 7


def _get_client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ---------------------------------------------------------------------------
# Signal extraction helpers
# ---------------------------------------------------------------------------

def _escalation_signals(
    item: Escalation,
    classification: Classification,
    entity: ResolvedEntity,
) -> tuple[list[str], list[str]]:
    supporting = []
    contradicting = []

    if item.severity in ("Critical", "High"):
        supporting.append(f"Severity: {item.severity}")
    if item.days_open > 14:
        supporting.append(f"Open {item.days_open} days")
    if item.escalations_last_30d >= 3:
        supporting.append(f"{item.escalations_last_30d} escalations in 30 days")
    if item.tier == "Enterprise":
        supporting.append("Enterprise tier")
    if item.arr_gbp > 100_000:
        supporting.append(f"ARR £{item.arr_gbp:,}")

    # Classification signals
    for signal in classification.urgency_indicators:
        if signal not in supporting:
            supporting.append(signal)

    if item.severity in ("Low", "Medium"):
        contradicting.append(f"Severity only {item.severity}")
    if item.escalations_last_30d < 2:
        contradicting.append("Low escalation frequency")

    return supporting, contradicting


def _opportunity_signals(
    item: Opportunity,
    classification: Classification,
    entity: ResolvedEntity,
) -> tuple[list[str], list[str]]:
    supporting = []
    contradicting = []

    if item.stage == "Closed Lost":
        supporting.append("Renewal already lost")
    if item.days_stalled > 14:
        supporting.append(f"Stalled {item.days_stalled} days")
    if item.tier == "Enterprise":
        supporting.append("Enterprise tier")
    if item.arr_gbp > 100_000:
        supporting.append(f"ARR £{item.arr_gbp:,}")

    for signal in classification.urgency_indicators:
        if signal not in supporting:
            supporting.append(signal)

    # Positive signals reduce urgency
    if classification.sentiment in ("positive", "mixed"):
        if "positive exec meeting" in item.notes.lower() or "strong intent" in item.notes.lower():
            contradicting.append("Positive executive engagement noted")
    if item.days_stalled == 0:
        contradicting.append("No stall — renewal on track")
    if item.stage not in ("Closed Lost", "Renewal"):
        contradicting.append(f"Stage '{item.stage}' not at immediate risk")

    return supporting, contradicting


def _project_signals(
    item: ProjectUpdate,
    classification: Classification,
    entity: ResolvedEntity,
) -> tuple[list[str], list[str]]:
    supporting = []
    contradicting = []

    if item.status == "Red":
        supporting.append("Project status Red")
    if item.days_overdue > 0:
        supporting.append(f"{item.days_overdue} days overdue")
    if item.budget_variance_pct > 10:
        supporting.append(f"Budget variance {item.budget_variance_pct}%")

    for signal in classification.urgency_indicators:
        if signal not in supporting:
            supporting.append(signal)

    if item.status == "Green":
        contradicting.append("Project status Green")
    if item.days_overdue == 0:
        contradicting.append("On schedule")
    if item.status == "Amber":
        contradicting.append("Amber status — monitoring, not critical")

    return supporting, contradicting


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _compute_score(
    supporting: list[str],
    contradicting: list[str],
    entity: ResolvedEntity,
    item: Escalation | Opportunity | ProjectUpdate,
) -> int:
    score = 0

    # Signals
    score += min(len(supporting), 4)
    score -= min(len(contradicting), 2)

    # Cross-stream corroboration
    if entity.cross_stream:
        score += 2

    # Structural bonuses — evaluated on the underlying item
    if isinstance(item, (Escalation, Opportunity)):
        if item.tier == "Enterprise":
            score += 1
        if item.arr_gbp > 100_000:
            score += 1

    if isinstance(item, Escalation):
        if item.days_open > 14:
            score += 1
        if item.escalations_last_30d >= 3:
            score += 1

    if isinstance(item, ProjectUpdate):
        if item.days_overdue > 0:
            score += 1
        if item.status == "Red":
            score += 1

    if isinstance(item, Opportunity):
        if item.stage == "Closed Lost":
            score += 1

    return max(0, min(10, score))


def _assign_risk_tier(score: int) -> str:
    if score >= 8:
        return "P1"
    if score >= 6:
        return "P2"
    if score >= 4:
        return "P3"
    return "P4"


def _assign_routing(risk_tier: str, score: int) -> str:
    if risk_tier in ("P1", "P2") and score >= EVIDENCE_AUTO_THRESHOLD:
        return "auto_approve"
    if risk_tier in ("P1", "P2") and score >= 4:
        return "human_review"
    return "monitor"


def _llm_rationale(
    item_summary: str,
    supporting: list[str],
    contradicting: list[str],
    score: int,
    entity: ResolvedEntity,
) -> str:
    client = _get_client()

    cross_note = (
        f" The entity appears across {len(entity.source_streams)} data streams "
        f"({', '.join(entity.source_streams)})."
        if entity.cross_stream
        else ""
    )

    prompt = (
        f"Evidence score: {score}/10.{cross_note}\n"
        f"Supporting signals: {', '.join(supporting) or 'none'}.\n"
        f"Contradicting signals: {', '.join(contradicting) or 'none'}.\n"
        f"Item summary: {item_summary}\n\n"
        "Write a single factual sentence (max 25 words) explaining what drives this evidence score."
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You write concise, factual rationale sentences for evidence scores.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_tokens=60,
    )
    return response.choices[0].message.content.strip()


def _recommended_action_and_owner(
    item: Escalation | Opportunity | ProjectUpdate,
    risk_tier: str,
    entity: ResolvedEntity,
) -> tuple[str, str]:
    """Derive recommended action and owner from item type and risk tier."""
    if isinstance(item, Escalation):
        if risk_tier == "P1":
            return "Immediate executive escalation — resolve authentication failure", "Customer Success"
        if risk_tier == "P2":
            return "Assign senior CSM — SLA breach risk", "Customer Success"
        return "Monitor — standard support channel", "Monitor"

    if isinstance(item, Opportunity):
        if item.stage == "Closed Lost":
            if risk_tier in ("P1", "P2"):
                return "Post-mortem and win-back strategy", "Sales Director"
        if risk_tier == "P1":
            return "Sales Director to engage new decision-maker", "Sales Director"
        if risk_tier == "P2":
            return "Account Executive to re-engage — renewal at risk", "Sales Director"
        return "Monitor renewal pipeline", "Monitor"

    if isinstance(item, ProjectUpdate):
        if risk_tier == "P1":
            return "CEO escalation — delivery risk with customer impact", "CEO"
        if risk_tier == "P2":
            return "PMO review — resourcing and timeline reset", "PMO"
        return "Monitor project health", "Monitor"

    return "No action required", "No Action"


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def score_items(
    escalations: list[Escalation],
    opportunities: list[Opportunity],
    projects: list[ProjectUpdate],
    classifications: list[Classification],
    entity_map: dict[str, ResolvedEntity],
) -> list[RiskAssessment]:
    """
    Score all items and return a RiskAssessment per item.
    Items without a resolved entity are skipped.
    """
    # Index classifications by item_id
    class_by_id: dict[str, Classification] = {c.item_id: c for c in classifications}

    all_items: list[Escalation | Opportunity | ProjectUpdate] = (
        list(escalations) + list(opportunities) + list(projects)
    )

    results: list[RiskAssessment] = []

    for item in all_items:
        entity = entity_map.get(item.id)
        if entity is None:
            continue  # unresolved entity — skip

        classification = class_by_id.get(item.id)
        if classification is None:
            continue

        # Extract signals
        if isinstance(item, Escalation):
            supporting, contradicting = _escalation_signals(item, classification, entity)
        elif isinstance(item, Opportunity):
            supporting, contradicting = _opportunity_signals(item, classification, entity)
        else:
            supporting, contradicting = _project_signals(item, classification, entity)

        # Compute deterministic score
        score = _compute_score(supporting, contradicting, entity, item)
        risk_tier = _assign_risk_tier(score)
        routing = _assign_routing(risk_tier, score)

        # LLM rationale (sentence only)
        summary = getattr(item, "issue_summary", None) or getattr(item, "notes", "")
        rationale = _llm_rationale(summary, supporting, contradicting, score, entity)

        # Action and owner
        action, owner = _recommended_action_and_owner(item, risk_tier, entity)

        evidence = EvidenceStrength(
            score=score,
            supporting=supporting,
            contradicting=contradicting,
            cross_stream=entity.cross_stream,
            rationale=rationale,
        )

        results.append(
            RiskAssessment(
                item_id=item.id,
                entity_id=entity.entity_id,
                risk_tier=risk_tier,  # type: ignore[arg-type]
                evidence=evidence,
                recommended_action=action,
                recommended_owner=owner,  # type: ignore[arg-type]
                routing=routing,  # type: ignore[arg-type]
            )
        )

    return results
