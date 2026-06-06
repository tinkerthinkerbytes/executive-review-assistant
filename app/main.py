"""
FastAPI service for executive-review-assistant.

Endpoints:
  POST /run                   — trigger pipeline, returns job_id
  GET  /status/{job_id}       — pipeline progress
  GET  /brief/{job_id}/stream — SSE stream of the executive brief
  POST /review/{item_id}      — submit human review decision
  GET  /health                — liveness
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from threading import Thread
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

from src.brief import build_brief_metadata, stream_brief
from src.classify import classify_all_sync
from src.ingest import load_all
from src.models import ReviewDecision, RiskAssessment
from src.resolve import resolve_entities
from src.review import load_review_log, partition_assessments, save_review_log, submit_api_review
from src.score import score_items

logging.basicConfig(
    level=logging.INFO,
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# In-memory job store (sufficient for portfolio / single-worker deployment)
# ---------------------------------------------------------------------------

class JobState:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status: str = "pending"  # pending | running | complete | error
        self.started_at: str = datetime.now(timezone.utc).isoformat()
        self.completed_at: str | None = None
        self.error: str | None = None
        self.auto_approved: list[RiskAssessment] = []
        self.needs_review: list[RiskAssessment] = []
        self.monitored: list[RiskAssessment] = []
        self.entity_names: dict[str, str] = {}
        self.entity_streams: dict[str, list[str]] = {}
        self.review_decisions: list[ReviewDecision] = []

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "auto_approved": len(self.auto_approved),
            "needs_review": len(self.needs_review),
            "monitored": len(self.monitored),
            "error": self.error,
        }


_jobs: dict[str, JobState] = {}

# Index for looking up which job owns an item_id (for /review)
_item_to_job: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Background pipeline runner
# ---------------------------------------------------------------------------

def _run_pipeline_background(job: JobState) -> None:
    try:
        job.status = "running"
        logger.info(f"Pipeline started job_id={job.job_id}")

        escalations, opportunities, projects = load_all()
        classifications = classify_all_sync(escalations, opportunities, projects)
        entity_map = resolve_entities(escalations, opportunities, projects)
        assessments = score_items(
            escalations, opportunities, projects, classifications, entity_map
        )
        auto_approved, needs_review, monitored = partition_assessments(assessments)

        job.auto_approved = auto_approved
        job.needs_review = needs_review
        job.monitored = monitored

        for entity in entity_map.values():
            job.entity_names[entity.entity_id] = entity.canonical_name
            job.entity_streams[entity.entity_id] = list(entity.source_streams)

        # Register item→job mapping
        for a in assessments:
            _item_to_job[a.item_id] = job.job_id

        job.review_decisions = load_review_log()
        job.status = "complete"
        job.completed_at = datetime.now(timezone.utc).isoformat()
        logger.info(
            f"Pipeline complete job_id={job.job_id} "
            f"auto={len(auto_approved)} review={len(needs_review)} monitor={len(monitored)}"
        )

    except Exception as exc:
        job.status = "error"
        job.error = str(exc)
        logger.error(f"Pipeline error job_id={job.job_id} error={exc}")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("executive-review-assistant starting")
    yield
    logger.info("executive-review-assistant stopping")


app = FastAPI(
    title="executive-review-assistant",
    description="Decision-support system: what needs management attention today?",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ReviewRequest(BaseModel):
    decision: str  # approve | modify | skip
    original_recommendation: str
    revised_recommendation: str | None = None
    reason: str | None = None
    reviewer_id: str = "api_reviewer"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/run")
def run_pipeline():
    """Trigger the pipeline. Returns job_id immediately; pipeline runs in background."""
    job_id = str(uuid.uuid4())[:8]
    job = JobState(job_id)
    _jobs[job_id] = job

    thread = Thread(target=_run_pipeline_background, args=(job,), daemon=True)
    thread.start()

    logger.info(f"Job created job_id={job_id}")
    return {"job_id": job_id, "status": "running"}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    """Return current pipeline progress for a job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job.to_dict()


@app.get("/brief/{job_id}/stream")
def stream_brief_endpoint(job_id: str):
    """
    SSE stream of the executive brief for a completed job.
    Returns 202 if the job is still running.
    """
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    if job.status != "complete":
        raise HTTPException(status_code=202, detail=f"Job status: {job.status}")

    # Merge latest review decisions before streaming
    job.review_decisions = load_review_log()

    def _generate():
        yield from stream_brief(
            job.auto_approved,
            job.needs_review,
            job.monitored,
            job.review_decisions,
            job.entity_names,
            job.entity_streams,
        )

    return StreamingResponse(_generate(), media_type="text/event-stream")


@app.post("/review/{item_id}")
def submit_review(item_id: str, body: ReviewRequest):
    """Submit a human review decision for an item flagged for review."""
    if body.decision not in ("approve", "modify", "skip"):
        raise HTTPException(status_code=400, detail="decision must be approve, modify, or skip")

    decision = submit_api_review(
        item_id=item_id,
        decision=body.decision,
        original_recommendation=body.original_recommendation,
        revised_recommendation=body.revised_recommendation,
        reason=body.reason,
        reviewer_id=body.reviewer_id,
    )

    logger.info(
        f"Review submitted item_id={item_id} decision={body.decision} reviewer={body.reviewer_id}"
    )
    return {
        "item_id": item_id,
        "decision": decision.decision,
        "timestamp": decision.timestamp.isoformat(),
    }
