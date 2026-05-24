"""API routes for the Video-to-PGN application."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse

from app.config import settings
from app.models.schemas import (
    ErrorResponse,
    GameResult,
    JobResult,
    ProcessRequest,
    ProcessResponse,
    ProcessingStage,
)
from app.tasks import (
    get_event_log,
    get_event_signal,
    get_job_result,
    get_job_status,
    start_processing,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["api"])


@router.post("/process", response_model=ProcessResponse)
async def process_video(request: ProcessRequest):
    """Start processing a YouTube video URL.

    Returns a job_id that can be used to track progress via SSE.
    """
    url = request.url.strip()

    # Basic YouTube URL validation
    if not any(domain in url for domain in ["youtube.com", "youtu.be"]):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")

    if not settings.gemini_api_key:
        raise HTTPException(
            status_code=500,
            detail="Gemini API key not configured. Set VTP_GEMINI_API_KEY environment variable.",
        )

    job_id = await start_processing(url)
    return ProcessResponse(job_id=job_id)


@router.get("/status/{job_id}")
async def stream_status(job_id: str, request: Request):
    """Stream real-time progress updates via Server-Sent Events (SSE)."""
    job = get_job_status(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    async def event_generator():
        last_sent_index = 0

        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                break

            # Send any new events
            events = get_event_log(job_id)
            while last_sent_index < len(events):
                event = events[last_sent_index]
                data = event.model_dump_json()
                yield f"data: {data}\n\n"
                last_sent_index += 1

                # Stop streaming if complete or error
                if event.stage in (ProcessingStage.COMPLETE, ProcessingStage.ERROR):
                    return

            # Wait for new events (with timeout to send keepalive)
            signal = get_event_signal(job_id)
            if signal:
                signal.clear()
                try:
                    await asyncio.wait_for(signal.wait(), timeout=15.0)
                except asyncio.TimeoutError:
                    # Send keepalive comment
                    yield ": keepalive\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/games/{job_id}", response_model=JobResult)
async def get_games(job_id: str):
    """Get all extracted games for a completed job."""
    result = get_job_result(job_id)
    if result is None:
        job = get_job_status(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if job.stage == ProcessingStage.ERROR:
            raise HTTPException(status_code=500, detail=job.error or "Processing failed")
        raise HTTPException(status_code=202, detail="Job still processing")
    return result


@router.get("/download/{job_id}/{game_num}")
async def download_game_pgn(job_id: str, game_num: int):
    """Download a single game's PGN file."""
    result = get_job_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Job not found or not complete")

    game = next((g for g in result.games if g.game_number == game_num), None)
    if game is None:
        raise HTTPException(status_code=404, detail=f"Game {game_num} not found")

    # Check for saved file
    pgn_path = settings.output_dir / job_id / f"game_{game_num}.pgn"
    if pgn_path.exists():
        return FileResponse(
            path=str(pgn_path),
            filename=f"game_{game_num}.pgn",
            media_type="application/x-chess-pgn",
        )

    # Fallback: return PGN string as file
    from fastapi.responses import Response
    return Response(
        content=game.pgn,
        media_type="application/x-chess-pgn",
        headers={"Content-Disposition": f'attachment; filename="game_{game_num}.pgn"'},
    )


@router.get("/download/{job_id}")
async def download_all_pgns(job_id: str):
    """Download all games as a combined PGN file."""
    result = get_job_result(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Job not found or not complete")

    # Combine all PGNs with double newline separator
    combined = "\n\n".join(game.pgn for game in result.games if game.pgn)
    
    if not combined:
        raise HTTPException(status_code=404, detail="No valid games were extracted to download.")

    from fastapi.responses import Response
    safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in result.video_title)[:50]
    filename = f"{safe_title or 'games'}_all.pgn"

    return Response(
        content=combined,
        media_type="application/x-chess-pgn",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
