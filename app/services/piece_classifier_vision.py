"""Piece classification via Gemini Vision API.

Sends full chess-video screenshots to Google's Gemini 2.5 Flash model and
parses the response into FEN piece-placement strings.
"""

from __future__ import annotations

import asyncio
import logging
import json
import cv2
import numpy as np
from typing import Optional

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# Coordinates of the active chess board squares in the 1280x720 frame.
# This aligns exactly with the wooden board square borders.
_BOARD_TOP = 32
_BOARD_BOTTOM = 680
_BOARD_LEFT = 32
_BOARD_RIGHT = 680

_MAX_RETRIES = 4
_INITIAL_BACKOFF = 4.0  # seconds


def _draw_grid_lines_with_labels(cropped_board) -> bytes:
    """Resize board to 800x800 and overlay green grid lines + tiny coordinate labels."""
    board = cv2.resize(cropped_board, (800, 800))
    grid_img = board.copy()
    
    cell_size = 100
    for r in range(8):
        for c in range(8):
            # Draw thin green grid lines
            if c < 7:
                cx = int((c + 1) * cell_size)
                cv2.line(grid_img, (cx, 0), (cx, 800), (0, 255, 0), 1)
            if r < 7:
                cy = int((r + 1) * cell_size)
                cv2.line(grid_img, (0, cy), (800, cy), (0, 255, 0), 1)
                
            # Draw tiny label like "R1C1" in the top-left of each cell
            x = int(c * cell_size + 4)
            y = int(r * cell_size + 12)
            label = f"R{r+1}C{c+1}"
            cv2.putText(grid_img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(grid_img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (220, 220, 220), 1, cv2.LINE_AA)
            
    _, encoded = cv2.imencode(".jpg", grid_img)
    return encoded.tobytes()


def _map_to_standard_fen(rows: list[list[str]], orientation: str) -> str:
    """Map the 8x8 list of piece descriptions to a standard FEN string."""
    standard_board = [["" for _ in range(8)] for _ in range(8)]
    
    mapping = {
        "White Pawn": "P", "White Knight": "N", "White Bishop": "B", 
        "White Rook": "R", "White Queen": "Q", "White King": "K",
        "Black Pawn": "p", "Black Knight": "n", "Black Bishop": "b", 
        "Black Rook": "r", "Black Queen": "q", "Black King": "k",
        "empty": ""
    }
    
    for r_idx in range(8):
        for c_idx in range(8):
            piece_desc = rows[r_idx][c_idx]
            piece_char = mapping.get(piece_desc, "")
            
            if orientation == "Black":
                # Flipped board (Black at bottom): Row 1 of image (r_idx=0) is Rank 1, Col 1 (c_idx=0) is File h
                rank = r_idx
                file = 7 - c_idx
            else:
                # Normal board (White at bottom): Row 1 of image (r_idx=0) is Rank 8, Col 1 (c_idx=0) is File a
                rank = 7 - r_idx
                file = c_idx
                
            standard_board[rank][file] = piece_char
            
    # Construct standard FEN representation (Rank 8 down to Rank 1)
    fen_rows = []
    for rank in reversed(range(8)):
        row_str = ""
        empty_count = 0
        for file in range(8):
            piece = standard_board[rank][file]
            if piece == "":
                empty_count += 1
            else:
                if empty_count > 0:
                    row_str += str(empty_count)
                    empty_count = 0
                row_str += piece
        if empty_count > 0:
            row_str += str(empty_count)
        fen_rows.append(row_str)
        
    return "/".join(fen_rows)


async def classify_frame(
    frame_path: str,
    client: genai.Client,
) -> Optional[str]:
    """Classify a single full frame using the grid-aligned 2-step FEN pipeline."""
    backoff = _INITIAL_BACKOFF
    
    # Load and crop the image to the exact active chess board coordinates
    img = cv2.imread(frame_path)
    if img is None:
        logger.error("Failed to read image path: %s", frame_path)
        return None
        
    cropped = img[_BOARD_TOP:_BOARD_BOTTOM, _BOARD_LEFT:_BOARD_RIGHT]
    
    # 1. Local pre-filter check: verify chess board grid is present
    try:
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        
        h_proj = np.sum(edges, axis=1)
        v_proj = np.sum(edges, axis=0)
        
        if np.std(h_proj) < 5000 or np.std(v_proj) < 5000:
            logger.info("Local grid check failed (no board visible) for %s", frame_path)
            return None
    except Exception as e:
        logger.warning("Local board grid check failed for %s: %s", frame_path, e)
    
    # Generate the grid-labeled image bytes
    labeled_bytes = _draw_grid_lines_with_labels(cropped)
    image_part = types.Part.from_bytes(data=labeled_bytes, mime_type='image/jpeg')
    
    prompt = """Look at this cropped chess board.
The board has a thin green grid separating the squares, and a tiny label like "RxCy" in the top-left of each square (where x is row 1..8 and y is column 1..8).
Analyze the board row-by-row from top-to-bottom and column-by-column from left-to-right (where Row 1 is the top-most row, and Row 8 is the bottom-most row).

Identify the pieces on each square of the 8x8 grid. If no chess board is visible at all (e.g. it is a splash screen, slide, menu, or face), set `is_board_visible` to false.

CRITICAL RULES:
1. Pay close attention to squares that are empty. If a piece has moved (e.g. a pawn has advanced from its starting square on row 2/7), its original starting square is now empty. Do NOT list the piece on both its starting square and its new square!
2. Look closely at the green lines to see exactly which square contains each piece.
"""

    schema = {
        "type": "OBJECT",
        "properties": {
            "is_board_visible": {
                "type": "BOOLEAN"
            },
            "bottom_pieces_color": {
                "type": "STRING",
                "enum": ["White", "Black", "None"]
            },
            "rows": {
                "type": "ARRAY",
                "items": {
                    "type": "ARRAY",
                    "items": {
                        "type": "STRING",
                        "enum": [
                            "empty",
                            "White Pawn", "White Knight", "White Bishop", "White Rook", "White Queen", "White King",
                            "Black Pawn", "Black Knight", "Black Bishop", "Black Rook", "Black Queen", "Black King"
                        ]
                    }
                }
            }
        },
        "required": ["is_board_visible", "bottom_pieces_color", "rows"]
    }

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = await asyncio.to_thread(
                client.models.generate_content,
                model="gemini-2.5-flash",
                contents=[image_part, prompt],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    max_output_tokens=1500,
                    response_mime_type="application/json",
                    response_schema=schema,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            
            data = json.loads(response.text)
            if not data.get("is_board_visible", False):
                logger.debug("No board visible in %s", frame_path)
                return None
                
            rows = data["rows"]
            orientation = data["bottom_pieces_color"]
            
            # Map 8x8 list to standard FEN
            fen = _map_to_standard_fen(rows, orientation)
            logger.debug("Classified %s -> %s", frame_path, fen)
            return fen

        except Exception as exc:
            exc_str = str(exc)
            is_rate_limit = "429" in exc_str or "exhausted" in exc_str.lower() or "limit" in exc_str.lower()
            
            logger.warning(
                "Attempt %d/%d failed for %s (rate_limit=%s): %s",
                attempt,
                _MAX_RETRIES,
                frame_path,
                is_rate_limit,
                exc,
            )

            if attempt < _MAX_RETRIES:
                sleep_time = 15.0 if is_rate_limit else backoff
                await asyncio.sleep(sleep_time)
                if not is_rate_limit:
                    backoff *= 2

    logger.error("All %d attempts exhausted for %s", _MAX_RETRIES, frame_path)
    return None


async def classify_frames(
    frame_paths: list[str],
    client: genai.Client,
    delay_between_calls: float = 1.0,
    progress_callback=None,
) -> list[Optional[str]]:
    """Classify multiple full video frames sequentially."""
    if not frame_paths:
        return []

    total = len(frame_paths)
    results: list[Optional[str]] = []

    logger.info("Classifying %d frames using grid-labeled FEN pipeline", total)

    for i, path in enumerate(frame_paths):
        fen = await classify_frame(path, client)
        results.append(fen)

        if progress_callback:
            progress_callback(i + 1, total)

        if i < total - 1:
            await asyncio.sleep(delay_between_calls)

    success = sum(1 for f in results if f is not None)
    logger.info("Classification complete: %d/%d frames had visible boards", success, total)
    return results
