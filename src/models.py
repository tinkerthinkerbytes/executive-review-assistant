"""
Pydantic models for all pipeline stages.
Input models → Stage 1 classification → Stage 2 entity resolution →
Stage 3 evidence scoring → Stage 4/5 routing and review → Stage 6 brief generation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------

class Escalation(BaseModel):
    id: str
    account_name: str
    tier: Literal["Enterprise", "Growth", "Starter"]
    arr_gbp: int
    issue_summary: str
    days_open: int
    escalations_last_30d: int
    severity: Literal["Critical", "High", "Medium", "Low"]
    assigned_owner: str


class Opportunity(BaseModel):
    id: str
    account_name: str
    tier: Literal["Enterprise", "Growth", "Starter"]
    arr_gbp: int
    stage: Literal["Renewal", "Upsell", "Closed Lost", "New Business"]
    renewal_date: str
    days_stalled: int
    deal_value_gbp: int
    notes: str
    owner: str


class ProjectUpdate(BaseModel):
    id: str
    account_name: str
    project_name: str
    status: Literal["Red", "Amber", "Green"]
    phase: str
    days_overdue: int
    budget_variance_pct: int
    notes: str
    recommended_action: str
    owner: str


# ---------------------------------------------------------------------------
# Stage 1 — Classification
# ---------------------------------------------------------------------------

class Classification(BaseModel):
    item_id: str
    item_type: Literal["escalation", "opportunity", "project"]
    key_signals: list[str] = Field(
        description="2-4 concise signals that drive urgency or risk, drawn from the item content"
    )
    sentiment: Literal["positive", "neutral", "negative", "mixed"]
    urgency_indicators: list[str] = Field(
        description="Explicit time pressures, deadlines, or escalation flags present in the item"
    )


# ---------------------------------------------------------------------------
# Stage 2 — Entity Resolution
# ---------------------------------------------------------------------------

class ResolvedEntity(BaseModel):
    entity_id: str  # e.g. ACME_001
    canonical_name: str  # e.g. "Acme Corporation"
    aliases: list[str]  # all raw names that resolved to this entity
    source_streams: list[Literal["escalation", "opportunity", "project"]]
    cross_stream: bool  # True if entity appears in 2+ streams


# ---------------------------------------------------------------------------
# Stage 3 — Evidence Scoring
# ---------------------------------------------------------------------------

class EvidenceStrength(BaseModel):
    score: int = Field(ge=0, le=10, description="Rules-based evidence score 0-10")
    supporting: list[str] = Field(description="Signals that increase urgency/risk")
    contradicting: list[str] = Field(description="Signals that reduce urgency/risk")
    cross_stream: bool
    rationale: str = Field(
        description="Single sentence LLM-written rationale for the score"
    )


class RiskAssessment(BaseModel):
    item_id: str
    entity_id: str
    risk_tier: Literal["P1", "P2", "P3", "P4"]
    evidence: EvidenceStrength
    recommended_action: str
    recommended_owner: Literal[
        "CEO", "Sales Director", "Customer Success", "PMO", "Monitor", "No Action"
    ]
    routing: Literal["auto_approve", "human_review", "monitor"]


# ---------------------------------------------------------------------------
# Stage 5 — Human Review
# ---------------------------------------------------------------------------

class ReviewDecision(BaseModel):
    item_id: str
    timestamp: datetime
    decision: Literal["approve", "modify", "skip"]
    original_recommendation: str
    revised_recommendation: str | None = None
    reason: str | None = None
    reviewer_id: str


# ---------------------------------------------------------------------------
# Stage 6 — Brief Generation
# ---------------------------------------------------------------------------

class BriefItem(BaseModel):
    entity_id: str
    canonical_name: str
    risk_tier: Literal["P1", "P2", "P3", "P4"]
    headline: str
    context: str
    action: str
    owner: str
    evidence_score: int
    source_streams: list[str]


class ExecutiveBrief(BaseModel):
    generated_at: datetime
    p1_items: list[BriefItem]
    p2_items: list[BriefItem]
    p3_watch: list[BriefItem]
    items_reviewed: int
    items_auto_approved: int
    items_human_reviewed: int
    items_monitored: int
