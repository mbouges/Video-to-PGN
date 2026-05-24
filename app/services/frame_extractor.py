"""Frame extraction service using OpenCV.

Samples full video frames at a fixed interval (e.g. every 3 seconds) and
uses perceptual hashing to skip visually identical consecutive frames.
Saves frames as JPEG to minimise file size for API uploads.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class SampledFrame:
    """A sampled video frame."""

    frame_path: str
    timestamp: float  # seconds into the video
    frame_number: int


class FrameExtractionError(Exception):
    """Raised when frame extraction fails."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _open_video(video_path: str) -> cv2.VideoCapture:
    """Open a video file and return the capture object, or raise."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FrameExtractionError(f"Cannot open video: {video_path}")
    return cap


def _phash(image: np.ndarray, hash_size: int = 16) -> np.ndarray:
    """Compute a simple perceptual hash for deduplication.

    Resizes the image to a small square, converts to grayscale,
    and returns a binary hash based on the DCT.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    mean_val = float(np.mean(resized))
    return (resized > mean_val).flatten()


def _hamming_distance(h1: np.ndarray, h2: np.ndarray) -> int:
    """Hamming distance between two binary hash arrays."""
    return int(np.sum(h1 != h2))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_sampled_frames(
    video_path: str,
    output_dir: str,
    interval_seconds: float = 3.0,
    dedup_threshold: int = 20,
) -> list[SampledFrame]:
    """Extract full video frames at a fixed time interval.

    Parameters
    ----------
    video_path:
        Path to the input video file.
    output_dir:
        Directory to write extracted frame images (JPEG).
    interval_seconds:
        Time between sampled frames in seconds.
    dedup_threshold:
        Maximum perceptual hash hamming distance for two frames to be
        considered identical.  Frames that match the previous frame are
        skipped.

    Returns
    -------
    list[SampledFrame]
        Chronologically ordered sampled frames.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = _open_video(video_path)

    video_fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0
    frame_interval = max(1, int(round(video_fps * interval_seconds)))

    logger.info(
        "Sampling full frames every %.1fs (every %d frames) from %s "
        "(%.0fs duration, %d total frames)",
        interval_seconds,
        frame_interval,
        video_path,
        duration,
        total_frames,
    )

    # Board crop coordinates for deduplication
    board_top, board_bottom = 32, 680
    board_left, board_right = 32, 680

    prev_hash: np.ndarray | None = None
    frames: list[SampledFrame] = []
    frame_idx = 0
    skipped = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % frame_interval == 0:
            # Crop to board area for more accurate perceptual hashing
            # (ignores moving elements like webcam overlays, clocks, chat)
            try:
                board_crop = frame[board_top:board_bottom, board_left:board_right]
                current_hash = _phash(board_crop)
            except Exception:
                current_hash = _phash(frame)

            if prev_hash is not None:
                dist = _hamming_distance(current_hash, prev_hash)
                if dist < dedup_threshold:
                    skipped += 1
                    frame_idx += 1
                    continue

            prev_hash = current_hash
            timestamp = frame_idx / video_fps

            fname = f"frame_{frame_idx:06d}.jpg"
            fpath = os.path.join(output_dir, fname)
            cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

            frames.append(SampledFrame(fpath, timestamp, frame_idx))

        frame_idx += 1

    cap.release()
    logger.info(
        "Extracted %d frames (%d skipped as duplicates) to %s",
        len(frames),
        skipped,
        output_dir,
    )
    return frames
