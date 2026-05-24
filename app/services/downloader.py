"""Video downloader service using yt-dlp.

Downloads YouTube videos at a specified quality using the yt-dlp Python API.
Runs the blocking download in a background thread via asyncio.to_thread so
the caller can ``await`` it without blocking the event loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

import yt_dlp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class DownloadResult:
    """Result returned after a successful video download."""

    file_path: str
    title: str
    duration_seconds: float
    video_id: str


class DownloadError(Exception):
    """Raised when a video download fails."""


# ---------------------------------------------------------------------------
# Quality helpers
# ---------------------------------------------------------------------------

_QUALITY_MAP: dict[str, str] = {
    "360p": "bestvideo[ext=mp4][height<=360]/best[ext=mp4][height<=360]",
    "480p": "bestvideo[ext=mp4][height<=480]/best[ext=mp4][height<=480]",
    "720p": "bestvideo[ext=mp4][height<=720]/best[ext=mp4][height<=720]",
    "1080p": "bestvideo[ext=mp4][height<=1080]/best[ext=mp4][height<=1080]",
    "best": "bestvideo[ext=mp4]/best[ext=mp4]",
}


def _format_for_quality(quality: str) -> str:
    """Return a yt-dlp format string for the requested quality label."""
    key = quality.lower().strip()
    if key in _QUALITY_MAP:
        return _QUALITY_MAP[key]
    # If the caller passed a raw yt-dlp format string, use it directly.
    return key


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------


def _make_progress_hook(
    callback: Callable[[float], None] | None = None,
) -> Callable[[dict[str, Any]], None]:
    """Return a yt-dlp progress hook that forwards percentage to *callback*."""

    def _hook(d: dict[str, Any]) -> None:
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate", 0)
            downloaded = d.get("downloaded_bytes", 0)
            if total:
                pct = downloaded / total * 100
                logger.debug("Download progress: %.1f%%", pct)
                if callback is not None:
                    callback(pct)
        elif d.get("status") == "finished":
            logger.info("Download finished, post-processing …")
            if callback is not None:
                callback(100.0)

    return _hook


# ---------------------------------------------------------------------------
# Core download logic (synchronous – called via asyncio.to_thread)
# ---------------------------------------------------------------------------


def _download_sync(
    url: str,
    output_dir: str,
    quality: str,
    progress_callback: Callable[[float], None] | None = None,
) -> DownloadResult:
    """Download a single YouTube video synchronously and return metadata."""

    os.makedirs(output_dir, exist_ok=True)

    outtmpl = os.path.join(output_dir, "%(title)s [%(id)s].%(ext)s")
    ydl_opts: dict[str, Any] = {
        "format": _format_for_quality(quality),
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [_make_progress_hook(progress_callback)],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info: dict[str, Any] = ydl.extract_info(url, download=True)
            if info is None:
                raise DownloadError(f"yt-dlp returned no info for {url}")
    except yt_dlp.utils.DownloadError as exc:
        raise DownloadError(f"Failed to download video: {exc}") from exc

    # Resolve the final file path written to disk.
    file_path: str | None = None
    if "requested_downloads" in info and info["requested_downloads"]:
        file_path = info["requested_downloads"][0].get("filepath")
    if file_path is None:
        # Fallback: reconstruct from template.
        file_path = ydl.prepare_filename(info)
        # yt-dlp may have merged to mp4.
        base, _ = os.path.splitext(file_path)
        mp4_path = base + ".mp4"
        if os.path.exists(mp4_path):
            file_path = mp4_path

    if not os.path.isfile(file_path):
        raise DownloadError(f"Downloaded file not found at {file_path}")

    return DownloadResult(
        file_path=file_path,
        title=info.get("title", "Unknown"),
        duration_seconds=float(info.get("duration", 0)),
        video_id=info.get("id", ""),
    )


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------


async def download_video(
    url: str,
    output_dir: str,
    quality: str = "720p",
    progress_callback: Callable[[float], None] | None = None,
) -> DownloadResult:
    """Download a YouTube video asynchronously.

    Parameters
    ----------
    url:
        YouTube video URL.
    output_dir:
        Directory to save the downloaded file into.
    quality:
        Desired video quality label (``360p``, ``480p``, ``720p``, ``1080p``,
        ``best``) or a raw yt-dlp format string.
    progress_callback:
        Optional callable receiving download progress as a float 0–100.

    Returns
    -------
    DownloadResult
        Metadata about the downloaded file.

    Raises
    ------
    DownloadError
        If the download fails for any reason.
    """
    logger.info("Starting download: %s (quality=%s)", url, quality)
    result = await asyncio.to_thread(
        _download_sync, url, output_dir, quality, progress_callback
    )
    logger.info("Downloaded '%s' -> %s", result.title, result.file_path)
    return result
