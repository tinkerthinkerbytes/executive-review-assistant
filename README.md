# executive-review-assistant

A decision-support system that answers one question: **what needs management attention today?**

Takes a daily batch of mixed operational signals — customer escalations, sales opportunities, project status updates — and produces a prioritised, evidence-backed executive briefing. Ambiguous cases route to human review before the brief is generated. The system handles messy real-world inputs: entity names that don't match across CRM, support, and project systems are resolved to canonical entities before any scoring occurs.

**Stack:** Python 3.11 · OpenAI `gpt-4.1-nano` · FastAPI · Pydantic v2 · rapidfuzz · asyncio · Docker

---

## Architecture

```
inputs/
├── escalations.json
├── opportunities.json
└── projects.json
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1 — Classify (async parallel)                    │
│  LLM per item → key_signals, sentiment, urgency         │
│  asyncio.gather — all items classified simultaneously   │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2 — Entity Resolution                            │
│  rapidfuzz token_sort_ratio → canonical entity_id       │
│  "Acme Ltd" + "ACME Corp" + "acme" → ACME_001           │
│  LLM disambiguation for ambiguous fuzzy matches only    │
│  Cross-stream membership recorded here                  │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 3 — Evidence Scoring                             │
│  Rules-based score 0–10 (deterministic)                 │
│  +signals, -contradictions, +2 cross-stream             │
│  +tier/ARR/recency bonuses                              │
│  LLM writes rationale sentence — does not produce score │
│  Risk tier P1–P4 assigned                               │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 4 — Routing                                      │
│  score ≥ 7 + P1/P2 → auto_approve                       │
│  score 4–6 + P1/P2 → human_review                       │
│  score < 4          → monitor                           │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 5 — Human Review Gate                            │
│  CLI: item summary, risk tier, evidence, signals        │
│  API: POST /review/{item_id}                            │
│  Decision stored: approve / modify / skip               │
│  review_log.json: captures overrides with rationale     │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 6 — Brief Generation (streaming)                 │
│  Approved P1/P2 items → executive brief                 │
│  OpenAI stream=True → SSE via FastAPI                   │
│  P1 headline/context/action · P2 items · P3 watch list  │
└─────────────────────────────────────────────────────────┘
```

---

## Setup

```bash
git clone https://github.com/tinkerthinkerbytes/executive-review-assistant
cd executive-review-assistant
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Set your API key:
```bash
# Windows PowerShell:
'OPENAI_API_KEY=sk-...' | Out-File -FilePath .env -Encoding utf8
# macOS/Linux:
echo "OPENAI_API_KEY=sk-..." > .env
```

---

## Usage

### CLI — full pipeline with interactive review
```bash
python -m src.runner
```
The pipeline runs all six stages, pauses at Stage 5 for human review of ambiguous items, then streams the executive brief to stdout.

### API — FastAPI service
```bash
uvicorn app.main:app --reload
```

```bash
# Trigger pipeline
curl -X POST http://localhost:8000/run

# Check status
curl http://localhost:8000/status/{job_id}

# Stream brief (after pipeline completes)
curl http://localhost:8000/brief/{job_id}/stream

# Submit review decision
curl -X POST http://localhost:8000/review/{item_id} \
  -H "Content-Type: application/json" \
  -d '{"decision": "modify", "original_recommendation": "CEO escalation", \
       "revised_recommendation": "Customer Success Manager outreach only", \
       "reason": "Internal resourcing issue, not customer dissatisfaction", \
       "reviewer_id": "ops_lead"}'
```

### Docker
```bash
docker build -t executive-review-assistant .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... executive-review-assistant
```

### Eval
```bash
python -m eval.runner --demo
python -m eval.runner --demo --gate  # exit 1 on failure
```

---

## Design Notes

**Entity resolution as a first-class stage.** The same account appears as "Acme Ltd" in the support system, "Acme Corporation" in the CRM, and "acme" in a follow-up ticket. Without resolution, these are three separate items with low individual scores. Resolved, they become one cross-stream entity with a corroboration bonus that correctly elevates the risk tier. This is the operational reality that naive AI demos miss.

**Evidence Strength, not LLM confidence.** The scoring function is deterministic and rules-based — supporting signals, contradicting signals, cross-stream corroboration, tier, ARR, recency. The LLM contributes exactly one sentence: a human-readable rationale. Scores are auditable and reproducible; the rationale is readable but non-authoritative. This separates "what the data says" from "what the model thinks."

**Cross-stream corroboration.** An entity appearing in escalations AND opportunities simultaneously is categorically more urgent than one appearing in only one stream. The +2 cross-stream bonus in the scoring function encodes this — it's the moment the system earns its value, converting three unrelated tickets into a coherent account risk picture.

**The deliberately bad recommendation.** `TECHCORP_001 / PROJ-001` scores 9/10 and auto-approves to "CEO escalation." The correct action — once a reviewer reads the project notes — is "Customer Success Manager outreach only," because the delay is caused by an internal resourcing gap, not customer dissatisfaction. This case is in the synthetic data by design. The override, captured in `review_log.json`, demonstrates why human review remains mandatory for consequential actions regardless of evidence score.

**Review log as institutional memory.** `review_log.json` records the original recommendation, the revised recommendation, the reason, the reviewer, and the timestamp for every override. Over time this log identifies systematic model errors, documents decision rationale for audit, and enables drift analysis — the same functions a change management system serves in production operations.

**Async parallel classification.** Stage 1 uses `asyncio.gather` to classify all items simultaneously. For 19 items this saves ~15 seconds of sequential latency. The pattern scales linearly — a real deployment processing hundreds of daily signals would see proportionally larger gains.

---

## Known Limitations

- **Evidence score is heuristic, not ground truth.** The scoring rules encode assumptions about what matters (tier, ARR, cross-stream presence). Different organisations would calibrate these differently. The score should be treated as a prioritisation signal, not a factual risk measurement.

- **Entity resolution will fail on heavily abbreviated or aliased names.** The fuzzy matching threshold (82) is tuned for the synthetic data. Production deployment would require calibration against real account name variants and a review process for unresolved entities.

- **Human review remains mandatory for ambiguous cases.** The routing threshold is configurable, but no threshold eliminates the need for human judgement on P1/P2 items with contradicting signals. The system is decision support, not decision automation.

- **External context is not available.** Recent calls, emails, relationship history, and internal commentary that exist outside the three input streams are invisible to the system. The TechCorp case demonstrates this explicitly — the correct override requires context that no structured data source captured.

- **Recommendation quality depends on input data quality.** Stale CRM records, missing escalation entries, and inconsistent project status reporting degrade the brief. The system makes the best decision possible given its inputs; it cannot compensate for upstream data quality failures.

---

## Input Data

| Stream | Items | Entity variants | Notes |
|---|---|---|---|
| `escalations.json` | 7 | 6 unique accounts | Includes duplicate ESC-006 (same account as ESC-001, different name) |
| `opportunities.json` | 7 | 6 unique accounts | Includes OPP-007 (duplicate CRM entry) and Closed Lost case |
| `projects.json` | 5 | 5 unique accounts | Includes deliberately bad recommendation case (PROJ-001) |

| Entity | Canonical name | Name variants | Streams | Case type |
|---|---|---|---|---|
| ACME_001 | Acme Corporation | "Acme Ltd", "Acme Corporation", "acme" | Escalation + Opportunity + Project | Easy P1 |
| APEX_001 | Apex Financial | "Apex Financial", "Apex Finance Ltd" | Escalation + Opportunity + Project | Secondary P1 |
| TECHCORP_001 | TechCorp Ltd | "Tech Corp Ltd", "TechCorp", "TechCorp Ltd" | Escalation + Opportunity + Project | Deliberately bad recommendation |
| MERIDIAN_001 | Meridian Group | "Meridian Group", "Meridian" | Escalation + Opportunity | Ambiguous / mixed signals |
| NOVA_001 | Nova Retail Ltd | "Nova Retail", "Nova Retail Ltd" | Escalation + Opportunity + Project | Easy P4 |
| SOLAR_001 | Solar Analytics | "Solar Analytics Co", "Solar Analytics" | Escalation + Opportunity + Project | Easy P4 |
