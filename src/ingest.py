"""
Stage 0 — Ingest: load and validate input JSON files into typed Pydantic models.
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import Escalation, Opportunity, ProjectUpdate

INPUTS_DIR = Path(__file__).parent.parent / "inputs"


def load_escalations(path: Path | None = None) -> list[Escalation]:
    p = path or INPUTS_DIR / "escalations.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    return [Escalation(**item) for item in raw]


def load_opportunities(path: Path | None = None) -> list[Opportunity]:
    p = path or INPUTS_DIR / "opportunities.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    return [Opportunity(**item) for item in raw]


def load_projects(path: Path | None = None) -> list[ProjectUpdate]:
    p = path or INPUTS_DIR / "projects.json"
    raw = json.loads(p.read_text(encoding="utf-8"))
    return [ProjectUpdate(**item) for item in raw]


def load_all(
    escalations_path: Path | None = None,
    opportunities_path: Path | None = None,
    projects_path: Path | None = None,
) -> tuple[list[Escalation], list[Opportunity], list[ProjectUpdate]]:
    """Load and validate all three input streams. Returns (escalations, opportunities, projects)."""
    escalations = load_escalations(escalations_path)
    opportunities = load_opportunities(opportunities_path)
    projects = load_projects(projects_path)
    return escalations, opportunities, projects
