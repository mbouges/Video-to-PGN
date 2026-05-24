"""Pydantic models for API requests, responses, and job tracking."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, HttpUrl


# --- Request Models ---


class ProcessRequest(BaseModel):
    """Request to start processing a YouTube video."""

    url: str = Field(..., description="YouTube video URL")


# --- Job Status ---


class ProcessingStage(str, Enum):
    """Stages of the video processing pipeline."""

    QUEUED = "queued"
    DOWNLOADING = "downloading"
    SEGMENTING = "segmenting"
    ANALYZING = "analyzing"
    GENERATING = "generating"
    COMPLETE = "complete"
    ERROR = "error"


class ProgressEvent(BaseModel):
    """A single progress update event sent via SSE."""

    stage: ProcessingStage
    progress: float = Field(0.0, ge=0.0, le=100.0, description="Progress percentage")
    message: str = ""
    current_game: Optional[int] = None
    total_games: Optional[int] = None


class JobStatus(BaseModel):
    """Current status of a processing job."""

    job_id: str
    stage: ProcessingStage
    progress: float = 0.0
    message: str = ""
    video_title: Optional[str] = None
    total_games: Optional[int] = None
    completed_games: int = 0
    error: Optional[str] = None


# --- Game Results ---


class GameResult(BaseModel):
    """Result of a single extracted chess game."""

    game_number: int
    result: str = "*"  # "1-0", "0-1", "1/2-1/2", or "*"
    moves_count: int = 0
    pgn: str = ""
    white: str = "?"
    black: str = "?"


class JobResult(BaseModel):
    """Complete results of a processing job."""

    job_id: str
    video_title: str = ""
    video_url: str = ""
    total_games: int = 0
    games: list[GameResult] = []


# --- API Responses ---


class ProcessResponse(BaseModel):
    """Response after starting a processing job."""

    job_id: str
    message: str = "Processing started"


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: Optional[str] = None
