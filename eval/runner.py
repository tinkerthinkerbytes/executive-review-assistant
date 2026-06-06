"""
Eval harness for executive-review-assistant.

Checks two properties after a pipeline run:
  1. completeness   — all expected entity IDs appear in the brief
  2. faithfulness   — LLM-as-judge: does the brief contain claims not supported by input signals?

Usage:
  python -m eval.runner [--gate] [--threshold 0.8]
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# Expected entities that must appear in a P1/P2 brief for this synthetic data
# ---------------------------------------------------------------------------

EXPECTED_P1_ENTITIES = ["ACME_001", "APEX_001"]
EXPECTED_P2_OR_HIGHER = ["TECHCORP_001"]  # may vary depending on review outcome

# ---------------------------------------------------------------------------
# LLM judge models
# ---------------------------------------------------------------------------

MODEL = "gpt-4.1-nano"


class FaithfulnessVerdict(BaseModel):
    faithful: bool
    unsupported_claims: list[str] = Field(default_factory=list)
    rationale: str


def _judge_faithfulness(brief_text: str, source_signals: list[str]) -> FaithfulnessVerdict:
    """Check whether the brief introduces claims not present in the source signals."""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    signals_text = "\n".join(f"- {s}" for s in source_signals)

    result = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are evaluating whether an executive brief is faithful to its source signals. "
                    "Identify any claims in the brief not supported by the provided signals."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Source signals:\n{signals_text}\n\n"
                    f"Brief text:\n{brief_text}\n\n"
                    "Is every claim in the brief grounded in the source signals? "
                    "List any unsupported claims."
                ),
            },
        ],
        response_format=FaithfulnessVerdict,
        temperature=0,
    )
    return result.choices[0].message.parsed


# ---------------------------------------------------------------------------
# Completeness check (deterministic)
# ---------------------------------------------------------------------------

def check_completeness(brief_text: str, expected_entities: list[str]) -> dict:
    hits = [e for e in expected_entities if e in brief_text or e.lower() in brief_text.lower()]
    missing = [e for e in expected_entities if e not in hits]
    return {
        "dimension": "completeness",
        "passed": len(missing) == 0,
        "score": len(hits) / len(expected_entities) if expected_entities else 1.0,
        "hits": hits,
        "missing": missing,
    }


# ---------------------------------------------------------------------------
# Main eval runner
# ---------------------------------------------------------------------------

def run_eval(brief_text: str, source_signals: list[str], gate: bool = False, threshold: float = 0.8) -> bool:
    print("\n" + "=" * 55)
    print("  EXECUTIVE REVIEW ASSISTANT — EVAL")
    print("=" * 55)

    all_passed = True

    # 1. Completeness
    all_expected = EXPECTED_P1_ENTITIES + EXPECTED_P2_OR_HIGHER
    completeness = check_completeness(brief_text, all_expected)
    mark = "✓" if completeness["passed"] else "✗"
    print(f"\n[{mark}] Completeness: {completeness['score']:.0%}")
    if completeness["missing"]:
        print(f"    Missing: {', '.join(completeness['missing'])}")
    if not completeness["passed"]:
        all_passed = False

    # 2. Faithfulness (LLM-as-judge)
    print("\n[~] Faithfulness: running LLM judge...", flush=True)
    faithfulness = _judge_faithfulness(brief_text, source_signals)
    mark = "✓" if faithfulness.faithful else "✗"
    print(f"[{mark}] Faithfulness: {'pass' if faithfulness.faithful else 'FAIL'}")
    print(f"    Rationale: {faithfulness.rationale}")
    if faithfulness.unsupported_claims:
        print("    Unsupported claims:")
        for c in faithfulness.unsupported_claims:
            print(f"      - {c}")
    if not faithfulness.faithful:
        all_passed = False

    print("\n" + "=" * 55)
    print(f"  RESULT: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 55 + "\n")

    if gate and not all_passed:
        sys.exit(1)

    return all_passed


def _demo_run():
    """Quick smoke-test against a dummy brief when called directly."""
    dummy_brief = (
        "ACME_001 (Acme Corporation) — P1: Critical authentication failure, 35 days open, "
        "5 escalations. Recommend immediate CEO escalation.\n"
        "APEX_001 (Apex Financial) — P1: Regulatory deadline in 11 days, audit trail gap. "
        "Executive sponsor engagement required.\n"
        "TECHCORP_001 (TechCorp Ltd) — P2: Analytics migration 8 days overdue."
    )
    signals = [
        "ACME_001: severity Critical, 35 days open, 5 escalations, Enterprise tier, ARR £185k",
        "APEX_001: regulatory deadline 11 days, audit trail missing, 21 days open, Enterprise £310k",
        "TECHCORP_001: project Red, 8 days overdue, internal resourcing gap",
    ]
    run_eval(dummy_brief, signals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval harness for executive-review-assistant")
    parser.add_argument("--gate", action="store_true", help="Exit 1 if eval fails")
    parser.add_argument("--threshold", type=float, default=0.8, help="Pass-rate threshold")
    parser.add_argument("--demo", action="store_true", help="Run against dummy data")
    args = parser.parse_args()

    if args.demo:
        _demo_run()
    else:
        print("Pass --demo to run against synthetic data, or import run_eval() directly.")
