"""Background task orchestrator for the video processing pipeline.

Coordinates: download → sample frames → classify positions → segment games
→ detect moves → build PGN.

Reports progress via an in-memory job store for SSE streaming.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path
from typing import Optional

from app.config import settings
from app.models.schemas import (
    GameResult,
    JobResult,
    JobStatus,
    ProcessingStage,
    ProgressEvent,
)
from app.services.downloader import download_video
from app.services.frame_extractor import extract_sampled_frames
from app.services.piece_classifier_vision import classify_frames
from app.services.move_detector import build_move_sequence
from app.services.pgn_builder import build_pgn, save_pgn, GameMetadata

logger = logging.getLogger(__name__)

# Standard starting position FEN (piece placement only)
_STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


# --- In-Memory Job Store ---

# Maps job_id -> JobStatus
_jobs: dict[str, JobStatus] = {}

# Maps job_id -> JobResult (when complete)
_results: dict[str, JobResult] = {}

# Maps job_id -> list of ProgressEvent (event log for SSE replay)
_events: dict[str, list[ProgressEvent]] = {}

# Maps job_id -> asyncio.Event (signals new events available)
_event_signals: dict[str, asyncio.Event] = {}


def get_job_status(job_id: str) -> Optional[JobStatus]:
    """Get the current status of a job."""
    return _jobs.get(job_id)


def get_job_result(job_id: str) -> Optional[JobResult]:
    """Get the completed result of a job."""
    return _results.get(job_id)


def get_event_log(job_id: str) -> list[ProgressEvent]:
    """Get all progress events for a job."""
    return _events.get(job_id, [])


def get_event_signal(job_id: str) -> Optional[asyncio.Event]:
    """Get the asyncio.Event signal for a job."""
    return _event_signals.get(job_id)


def _emit_event(job_id: str, event: ProgressEvent) -> None:
    """Record a progress event and update job status."""
    _events.setdefault(job_id, []).append(event)

    # Update job status
    if job_id in _jobs:
        job = _jobs[job_id]
        job.stage = event.stage
        job.progress = event.progress
        job.message = event.message
        if event.total_games is not None:
            job.total_games = event.total_games
        if event.current_game is not None:
            job.completed_games = event.current_game

    # Signal waiting SSE connections
    signal = _event_signals.get(job_id)
    if signal:
        signal.set()


async def start_processing(url: str) -> str:
    """Start processing a YouTube video. Returns job_id."""
    job_id = str(uuid.uuid4())[:8]

    # Initialize job tracking
    _jobs[job_id] = JobStatus(
        job_id=job_id,
        stage=ProcessingStage.QUEUED,
        message="Job queued",
    )
    _events[job_id] = []
    _event_signals[job_id] = asyncio.Event()

    # Launch background task
    asyncio.create_task(_process_video(job_id, url))

    return job_id


# ---------------------------------------------------------------------------
# Game segmentation from FEN sequence
# ---------------------------------------------------------------------------


def count_pieces(fen: str) -> int:
    """Count the total number of pieces on the board."""
    return sum(1 for c in fen.split()[0] if c.isalpha())


def get_piece_changes(fen1: str, fen2: str) -> int:
    """Calculate the number of squares that differ between two FENs."""
    def expand(f):
        board = []
        for row in f.split()[0].split('/'):
            for c in row:
                if c.isdigit():
                    board.extend([''] * int(c))
                else:
                    board.append(c)
        return board
    
    try:
        b1 = expand(fen1)
        b2 = expand(fen2)
        if len(b1) != 64 or len(b2) != 64:
            return 64
        return sum(1 for x, y in zip(b1, b2) if x != y)
    except Exception:
        return 64


def is_starting_like(fen: str) -> bool:
    """Check if the FEN is close to the starting chess position."""
    return get_piece_changes(fen, _STARTING_FEN) <= 6


def _debounce_fens(fens: list[Optional[str]]) -> list[Optional[str]]:
    """Remove classification noise/glitches using a 5-frame sliding window.

    Extended from 3-frame to 5-frame window so that runs of 2 bad consecutive
    AI reads (which happen with review/analysis frames) are also smoothed out.
    """
    if len(fens) < 3:
        return fens

    cleaned = list(fens)
    n = len(fens)

    # First pass: standard 3-frame glitch removal
    for i in range(1, n - 1):
        prev_f = cleaned[i - 1]
        curr_f = cleaned[i]
        next_f = cleaned[i + 1]

        if curr_f is None:
            # If it was temporarily None (hidden), but neighbors are the same and not None, restore it
            if prev_f is not None and next_f is not None and prev_f == next_f:
                cleaned[i] = prev_f
            continue

        # If curr_f is a FEN, but both neighbors are close to each other and different from curr_f
        if prev_f is not None and next_f is not None:
            diff_prev = get_piece_changes(prev_f, curr_f)
            diff_next = get_piece_changes(curr_f, next_f)
            diff_outer = get_piece_changes(prev_f, next_f)

            # If current frame is very different from both neighbors, but neighbors are close to each other
            if diff_prev > 10 and diff_next > 10 and diff_outer <= 4:
                cleaned[i] = prev_f

    # Second pass: 5-frame outlier removal
    # Replace frames that are far from the median of their 5-frame window
    for i in range(2, n - 2):
        curr_f = cleaned[i]
        if curr_f is None:
            continue

        window = [cleaned[j] for j in range(i - 2, i + 3) if cleaned[j] is not None]
        if len(window) < 4:
            continue

        # Count how many window frames are close to each other (within 4 squares)
        # Find the "consensus" position: the one closest to the most others
        diffs_to_others = []
        for j, w in enumerate(window):
            if w == curr_f:
                continue
            diffs_to_others.append(get_piece_changes(curr_f, w))

        if diffs_to_others and all(d > 12 for d in diffs_to_others):
            # curr_f is very different from all its neighbors — likely a bad classification
            # Replace with the previous clean frame
            if cleaned[i - 1] is not None:
                cleaned[i] = cleaned[i - 1]
                logger.debug("5-frame debounce: replaced outlier frame %d", i)

    return cleaned


def _segment_games_from_fens(
    fens: list[Optional[str]],
    timestamps: list[float],
) -> list[dict]:
    """Split a FEN sequence into individual games.

    Uses starting position resets, transition hidden states, and massive position changes
    to segment games without splitting on thinking timeline gaps or minor noise.
    """
    games: list[dict] = []
    current_segment: list[tuple[str, float]] = []
    board_was_hidden = False
    
    for fen, ts in zip(fens, timestamps):
        if fen is None:
            board_was_hidden = True
            continue
            
        start_new = False
        
        if not current_segment:
            start_new = True
        else:
            prev_fen, prev_ts = current_segment[-1]
            
            # Condition 1: Current FEN is close to starting position, resetting the game
            if is_starting_like(fen):
                if not is_starting_like(prev_fen):
                    start_new = True
            
            # Condition 2: Board was hidden and is now visible with a different position
            elif board_was_hidden and get_piece_changes(prev_fen, fen) > 10:
                start_new = True
                
            # Condition 3: Massive position changes indicating a new game setup
            elif get_piece_changes(prev_fen, fen) > 16:
                start_new = True
                
        board_was_hidden = False
        
        if start_new:
            # Close previous segment
            if len(current_segment) >= 3:
                games.append({
                    "fens": [f for f, _ in current_segment],
                    "start_time": current_segment[0][1],
                    "end_time": current_segment[-1][1],
                })
            current_segment = [(fen, ts)]
        else:
            current_segment.append((fen, ts))
            
    # Close trailing segment
    if len(current_segment) >= 3:
        games.append({
            "fens": [f for f, _ in current_segment],
            "start_time": current_segment[0][1],
            "end_time": current_segment[-1][1],
        })
        
    return games


def _merge_adjacent_segments(games: list[dict], all_fens: list[Optional[str]], timestamps: list[float]) -> list[dict]:
    """Merge segments that were incorrectly split by brief AI classification errors.

    Two adjacent segments are merged when:
    - They are separated by <= 5 null/gap frames in the original FEN sequence
    - The end of segment N is within 12 squares of the start of segment N+1
      (i.e., the board position is plausibly continuous)
    - Neither segment starts from the standard starting position
      (if seg N+1 starts from opening, it's a genuine new game)

    This repairs the common failure where 2-3 consecutive bad AI reads cause
    the segmenter to treat one game as two halves.
    """
    if len(games) <= 1:
        return games

    # Build a timestamp->index lookup for fast gap calculation
    ts_to_idx = {ts: i for i, ts in enumerate(timestamps)}

    merged = []
    i = 0
    while i < len(games):
        current = games[i]
        if i + 1 >= len(games):
            merged.append(current)
            i += 1
            continue

        next_seg = games[i + 1]

        # Don't merge if next segment starts from the standard opening
        # (that's a genuine new game boundary)
        if is_starting_like(next_seg["fens"][0]):
            merged.append(current)
            i += 1
            continue

        # Calculate the gap between the two segments (frames between end_time and start_time)
        end_ts = current["end_time"]
        start_ts = next_seg["start_time"]
        gap_seconds = start_ts - end_ts

        # Only merge if the gap is small (within ~15 seconds = ~5 frames at 3s sampling)
        if gap_seconds > 15:
            merged.append(current)
            i += 1
            continue

        # Check positional continuity: end of current segment vs start of next
        end_fen = current["fens"][-1]
        start_fen = next_seg["fens"][0]
        position_diff = get_piece_changes(end_fen, start_fen)

        if position_diff <= 12:
            # Merge the two segments
            combined_fens = current["fens"] + next_seg["fens"]
            merged_seg = {
                "fens": combined_fens,
                "start_time": current["start_time"],
                "end_time": next_seg["end_time"],
            }
            logger.info(
                "Merged segments: %.0fs-%.0fs + %.0fs-%.0fs (gap=%.0fs, pos_diff=%d)",
                current["start_time"], current["end_time"],
                next_seg["start_time"], next_seg["end_time"],
                gap_seconds, position_diff,
            )
            games[i + 1] = merged_seg  # Replace next with merged, skip current
            i += 1
            continue

        merged.append(current)
        i += 1

    return merged


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


async def _process_video(job_id: str, url: str) -> None:
    """Main processing pipeline — runs as a background task."""
    try:
        settings.ensure_dirs()
        job_dir = settings.temp_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = job_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        output_dir = settings.output_dir / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Stage 1: Download Video ---
        _emit_event(job_id, ProgressEvent(
            stage=ProcessingStage.DOWNLOADING,
            progress=0.0,
            message="Downloading video...",
        ))

        download_result = await download_video(
            url=url,
            output_dir=str(job_dir),
            progress_callback=lambda p: _emit_event(job_id, ProgressEvent(
                stage=ProcessingStage.DOWNLOADING,
                progress=p,
                message=f"Downloading video... {p:.0f}%",
            )),
        )

        _jobs[job_id].video_title = download_result.title
        logger.info(
            f"[{job_id}] Downloaded: {download_result.title} "
            f"({download_result.duration_seconds:.0f}s)"
        )

        # --- Stage 2: Extract Sampled Frames ---
        _emit_event(job_id, ProgressEvent(
            stage=ProcessingStage.SEGMENTING,
            progress=0.0,
            message="Extracting video frames...",
        ))

        sampled_frames = extract_sampled_frames(
            video_path=download_result.file_path,
            output_dir=str(frames_dir),
            interval_seconds=settings.sample_interval_seconds,
            dedup_threshold=settings.dedup_threshold,
        )

        logger.info(f"[{job_id}] Extracted {len(sampled_frames)} frames")

        _emit_event(job_id, ProgressEvent(
            stage=ProcessingStage.SEGMENTING,
            progress=100.0,
            message=f"Extracted {len(sampled_frames)} frames",
        ))

        if len(sampled_frames) < 2:
            raise RuntimeError(
                "Too few frames extracted from video. "
                "The video may be too short or contain no changes."
            )

        # --- Stage 3: Classify All Frames ---
        _emit_event(job_id, ProgressEvent(
            stage=ProcessingStage.ANALYZING,
            progress=0.0,
            message=f"Analyzing positions (0/{len(sampled_frames)})...",
        ))

        from google import genai
        gemini_client = genai.Client(api_key=settings.gemini_api_key)

        frame_paths = [f.frame_path for f in sampled_frames]
        timestamps = [f.timestamp for f in sampled_frames]

        def on_classify_progress(completed: int, total: int):
            pct = completed / total * 100
            _emit_event(job_id, ProgressEvent(
                stage=ProcessingStage.ANALYZING,
                progress=pct,
                message=f"Analyzing positions ({completed}/{total})...",
            ))

        all_fens = await classify_frames(
            frame_paths=frame_paths,
            client=gemini_client,
            delay_between_calls=settings.api_call_delay,
            progress_callback=on_classify_progress,
        )

        # Save raw FEN classifications for debugging
        try:
            import json
            fens_path = output_dir / "fens.json"
            with open(fens_path, "w") as f:
                json.dump({"timestamps": timestamps, "fens": all_fens}, f, indent=2)
            logger.info(f"[{job_id}] Saved raw FEN classifications to {fens_path}")
        except Exception as err:
            logger.warning(f"[{job_id}] Failed to save FEN classifications: {err}")

        # Count how many frames had visible boards
        visible_count = sum(1 for f in all_fens if f is not None)
        logger.info(
            f"[{job_id}] Classification done: "
            f"{visible_count}/{len(all_fens)} frames had boards"
        )

        if visible_count < 2:
            raise RuntimeError(
                "Could not detect a chess board in the video frames. "
                "The board may not be visible or the video format is unsupported."
            )

        # Debounce/smooth FENs to eliminate single-frame classification glitches
        debounced_fens = _debounce_fens(all_fens)

        # --- Stage 4: Segment into Games & Build PGNs ---
        game_segments = _segment_games_from_fens(debounced_fens, timestamps)
        logger.info(f"[{job_id}] Found {len(game_segments)} raw segment(s) before merging")

        # Merge segments that were split by brief AI classification errors
        game_segments = _merge_adjacent_segments(game_segments, debounced_fens, timestamps)
        total_games = len(game_segments)
        logger.info(f"[{job_id}] Found {total_games} segment(s) after merging, before filtering")

        all_game_results: list[GameResult] = []

        for i, game_data in enumerate(game_segments):
            game_num = i + 1
            game_fens = game_data["fens"]

            _emit_event(job_id, ProgressEvent(
                stage=ProcessingStage.ANALYZING,
                progress=90 + (game_num / total_games * 10),
                message=f"Building PGN for game {game_num}/{total_games}...",
                current_game=game_num,
                total_games=total_games,
            ))

            # Detect moves from FEN sequence
            move_sequence = build_move_sequence(game_fens)
            n_moves = len(move_sequence.moves)
            n_errors = len(move_sequence.errors)
            starts_from_opening = is_starting_like(game_fens[0])

            logger.info(
                f"[{job_id}] Segment {game_num}/{total_games}: "
                f"{len(game_fens)} positions, {n_moves} moves, {n_errors} errors, "
                f"time={game_data['start_time']:.0f}s-{game_data['end_time']:.0f}s, "
                f"starts_from_opening={starts_from_opening}"
            )

            # 1. Skip segments that are too short (likely reviews or static noise)
            # Relaxed from 4 to 2: even short game clips may be valid.
            if n_moves < 2:
                logger.info(
                    f"[{job_id}] SKIP Segment {game_num}: too few moves ({n_moves}). "
                    f"Likely a review or static frame."
                )
                continue

            # 2. Skip segments with overwhelmingly high error-to-move ratio.
            # Relaxed from 1:1 to 2:1 (errors must be 2x moves to reject).
            # Given AI noise, some errors in a real game are expected.
            if n_errors >= 2 * n_moves and n_moves < 5:
                logger.info(
                    f"[{job_id}] SKIP Segment {game_num}: too many errors "
                    f"({n_errors} errors vs {n_moves} moves). "
                    f"Likely a review/analysis jump or mirrored board."
                )
                continue

            # Build PGN
            # Determine game result from final board state
            try:
                board_result = move_sequence.board.result()
            except Exception:
                board_result = "*"

            game_number = len(all_game_results) + 1

            metadata = GameMetadata(
                event=download_result.title or "YouTube Video",
                site="Chess.com",
                date="????.??.??",
                round_num=str(game_number),
                white="?",
                black="?",
                result=board_result if board_result != "*" else "*",
                source_url=url,
            )

            pgn_string = build_pgn(move_sequence.moves, metadata)

            # Save individual PGN file
            pgn_path = output_dir / f"game_{game_number}.pgn"
            save_pgn(pgn_string, str(pgn_path))

            game_result = GameResult(
                game_number=game_number,
                result=metadata.result,
                moves_count=len(move_sequence.moves),
                pgn=pgn_string,
            )
            all_game_results.append(game_result)

            logger.info(
                f"[{job_id}] ACCEPT Segment {game_num} as Game {game_number}: "
                f"{n_moves} moves, {n_errors} errors, result={metadata.result}"
            )

        if not all_game_results:
            raise RuntimeError(
                f"Detected {visible_count} board positions across {total_games} segment(s) "
                f"but could not identify any complete games after filtering. "
                f"Check the server logs for per-segment move/error counts to diagnose."
            )

        # --- Stage 5: Complete ---
        result = JobResult(
            job_id=job_id,
            video_title=download_result.title or "",
            video_url=url,
            total_games=len(all_game_results),
            games=all_game_results,
        )
        _results[job_id] = result

        _emit_event(job_id, ProgressEvent(
            stage=ProcessingStage.COMPLETE,
            progress=100.0,
            message=f"Done! Extracted {len(all_game_results)} game(s)",
            total_games=len(all_game_results),
        ))

        logger.info(
            f"[{job_id}] Processing complete: "
            f"{len(all_game_results)} game(s) extracted"
        )

    except Exception as e:
        logger.exception(f"[{job_id}] Processing failed: {e}")
        _emit_event(job_id, ProgressEvent(
            stage=ProcessingStage.ERROR,
            progress=0.0,
            message=f"Error: {str(e)}",
        ))
        if job_id in _jobs:
            _jobs[job_id].error = str(e)
