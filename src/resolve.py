"""
Stage 2 — Entity Resolution: fuzzy match raw account names to canonical entity IDs.

Strategy:
  1. Exact match against known aliases (case-insensitive).
  2. rapidfuzz token_sort_ratio for approximate matches (threshold: 82).
  3. LLM disambiguation only for cases where fuzzy score falls in the ambiguous band (60–81).

Returns a dict mapping item_id → ResolvedEntity.
"""

from __future__ import annotations

import os
from typing import Union

from openai import OpenAI
from pydantic import BaseModel
from rapidfuzz import fuzz

from .models import Escalation, Opportunity, ProjectUpdate, ResolvedEntity

MODEL = "gpt-4.1-nano"

# ---------------------------------------------------------------------------
# Entity catalogue — source of truth for canonical names and seed aliases.
# Aliases are case-insensitive; new aliases discovered during resolution are
# added at runtime but not persisted.
# ---------------------------------------------------------------------------

ENTITY_CATALOGUE: list[dict] = [
    {
        "entity_id": "ACME_001",
        "canonical_name": "Acme Corporation",
        "aliases": ["acme ltd", "acme corporation", "acme corp", "acme"],
    },
    {
        "entity_id": "TECHCORP_001",
        "canonical_name": "TechCorp Ltd",
        "aliases": ["techcorp ltd", "techcorp", "tech corp ltd", "tech corp"],
    },
    {
        "entity_id": "MERIDIAN_001",
        "canonical_name": "Meridian Group",
        "aliases": ["meridian group", "meridian"],
    },
    {
        "entity_id": "APEX_001",
        "canonical_name": "Apex Financial",
        "aliases": ["apex financial", "apex finance ltd", "apex finance"],
    },
    {
        "entity_id": "NOVA_001",
        "canonical_name": "Nova Retail Ltd",
        "aliases": ["nova retail", "nova retail ltd"],
    },
    {
        "entity_id": "SOLAR_001",
        "canonical_name": "Solar Analytics",
        "aliases": ["solar analytics", "solar analytics co", "solar analytics ltd"],
    },
]

FUZZY_AUTO_THRESHOLD = 82   # >= this → auto-resolve
FUZZY_AMBIGUOUS_LOW = 60    # [60, 81] → LLM disambiguation
# < 60 → unresolved (new entity)


def _best_fuzzy_match(name: str) -> tuple[dict | None, int]:
    """Return (catalogue_entry, best_score) for the closest match."""
    name_lower = name.lower()
    best_entry = None
    best_score = 0

    for entry in ENTITY_CATALOGUE:
        for alias in entry["aliases"]:
            score = fuzz.token_sort_ratio(name_lower, alias)
            if score > best_score:
                best_score = score
                best_entry = entry

    return best_entry, best_score


def _llm_disambiguate(raw_name: str, candidates: list[dict]) -> dict | None:
    """Use LLM to pick the most likely canonical entity for an ambiguous name."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    candidate_text = "\n".join(
        f"- {e['entity_id']}: {e['canonical_name']} (known aliases: {', '.join(e['aliases'])})"
        for e in candidates
    )

    class DisambiguationResult(BaseModel):
        entity_id: str | None
        reasoning: str

    result = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are resolving company name variants to canonical entities. "
                    "Pick the most likely match, or return null if none is a plausible match."
                ),
            },
            {
                "role": "user",
                "content": (
                    f'Raw name: "{raw_name}"\n\nCandidates:\n{candidate_text}\n\n'
                    "Return the entity_id of the best match, or null if none apply."
                ),
            },
        ],
        response_format=DisambiguationResult,
        temperature=0,
    )

    parsed = result.choices[0].message.parsed
    if parsed.entity_id:
        for entry in candidates:
            if entry["entity_id"] == parsed.entity_id:
                return entry
    return None


def resolve_name(raw_name: str) -> dict | None:
    """Resolve a single raw account name to a catalogue entry, or None if unresolved."""
    # Step 1: exact alias match
    raw_lower = raw_name.lower().strip()
    for entry in ENTITY_CATALOGUE:
        if raw_lower in entry["aliases"]:
            return entry

    # Step 2: fuzzy match
    best_entry, best_score = _best_fuzzy_match(raw_name)

    if best_score >= FUZZY_AUTO_THRESHOLD:
        # Add alias for future exact matches
        if raw_lower not in best_entry["aliases"]:
            best_entry["aliases"].append(raw_lower)
        return best_entry

    if FUZZY_AMBIGUOUS_LOW <= best_score < FUZZY_AUTO_THRESHOLD:
        # Gather all candidates in the ambiguous band
        candidates = []
        for entry in ENTITY_CATALOGUE:
            for alias in entry["aliases"]:
                score = fuzz.token_sort_ratio(raw_lower, alias)
                if score >= FUZZY_AMBIGUOUS_LOW:
                    if entry not in candidates:
                        candidates.append(entry)
                    break
        return _llm_disambiguate(raw_name, candidates)

    return None  # unresolved


InputItem = Union[Escalation, Opportunity, ProjectUpdate]


def _item_stream(item: InputItem) -> str:
    if isinstance(item, Escalation):
        return "escalation"
    if isinstance(item, Opportunity):
        return "opportunity"
    return "project"


def resolve_entities(
    escalations: list[Escalation],
    opportunities: list[Opportunity],
    projects: list[ProjectUpdate],
) -> dict[str, ResolvedEntity]:
    """
    Resolve all input items to canonical entities.

    Returns a dict mapping item_id → ResolvedEntity.
    Each ResolvedEntity tracks which streams that entity appears in and
    whether it spans multiple streams (cross_stream=True).
    """
    # entity_id → accumulated data
    entity_map: dict[str, dict] = {}

    def _record(item_id: str, raw_name: str, stream: str) -> str | None:
        entry = resolve_name(raw_name)
        if entry is None:
            return None

        eid = entry["entity_id"]
        if eid not in entity_map:
            entity_map[eid] = {
                "entity_id": eid,
                "canonical_name": entry["canonical_name"],
                "aliases": set(),
                "source_streams": set(),
                "item_ids": [],
            }

        entity_map[eid]["aliases"].add(raw_name)
        entity_map[eid]["source_streams"].add(stream)
        entity_map[eid]["item_ids"].append(item_id)
        return eid

    all_items: list[tuple[str, str, str]] = (
        [(e.id, e.account_name, "escalation") for e in escalations]
        + [(o.id, o.account_name, "opportunity") for o in opportunities]
        + [(p.id, p.account_name, "project") for p in projects]
    )

    # item_id → entity_id lookup (for downstream stages)
    item_to_entity: dict[str, str] = {}
    for item_id, raw_name, stream in all_items:
        eid = _record(item_id, raw_name, stream)
        if eid:
            item_to_entity[item_id] = eid

    # Build ResolvedEntity objects keyed by item_id
    result: dict[str, ResolvedEntity] = {}
    for item_id, eid in item_to_entity.items():
        data = entity_map[eid]
        streams = list(data["source_streams"])
        result[item_id] = ResolvedEntity(
            entity_id=eid,
            canonical_name=data["canonical_name"],
            aliases=sorted(data["aliases"]),
            source_streams=streams,  # type: ignore[arg-type]
            cross_stream=len(data["source_streams"]) > 1,
        )

    return result
