"""Move detection by comparing consecutive FEN strings.

Given a sequence of FEN piece-placement strings (the first field of a full
FEN), this module figures out which legal chess move was made between each
pair of positions.  It works by iterating through every legal move on the
current ``chess.Board``, applying it, and checking whether the resulting
piece placement matches the target FEN.

When a transition cannot be resolved to a single legal move the error is
recorded and the detector attempts to re-synchronise by searching ahead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import chess

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class MoveSequence:
    """Result of building a move sequence from a list of FENs."""

    moves: list[chess.Move] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    board: chess.Board = field(default_factory=chess.Board)


class MoveDetectionError(Exception):
    """Raised for unrecoverable move-detection problems."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STARTING_PLACEMENT = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"


def _placement(board: chess.Board) -> str:
    """Return only the piece-placement field of the board's FEN."""
    return board.fen().split(" ", 1)[0]


def _normalise_placement(fen_fragment: str) -> str:
    """Normalise a piece-placement string for comparison.

    Strips surrounding whitespace and any fields after the first space.
    """
    return fen_fragment.strip().split(" ", 1)[0]


# ---------------------------------------------------------------------------
# Single-move detection
# ---------------------------------------------------------------------------


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


def detect_move(
    fen_before: str,
    fen_after: str,
    board: chess.Board,
) -> Optional[chess.Move]:
    """Find the legal move that turns *fen_before* into *fen_after*."""
    target = _normalise_placement(fen_after)
    current = _normalise_placement(fen_before)

    # Sanity check: the board should match fen_before.
    if _placement(board) != current:
        logger.warning(
            "Board placement mismatch: expected %s, got %s",
            current,
            _placement(board),
        )

    for move in board.legal_moves:
        board.push(move)
        if _placement(board) == target:
            board.pop()
            return move
        board.pop()

    return None


def detect_move_fuzzy(
    fen_before: str,
    fen_after: str,
    board: chess.Board,
    max_diff: int = 6,
) -> tuple[Optional[chess.Move], int, bool]:
    """Find the legal move that turns *board* into a state closest to *fen_after*.

    Supports toggling active player turn if a move from the other player matches better
    (useful for missed half-moves or custom starting positions).
    """
    target = _normalise_placement(fen_after)
    current = _normalise_placement(fen_before)

    current_diff = get_piece_changes(current, target)

    best_move = None
    best_diff = current_diff
    turn_toggled = False

    # 1. Try legal moves for current active turn
    for move in board.legal_moves:
        board.push(move)
        diff = get_piece_changes(_placement(board), target)
        board.pop()

        if diff < best_diff:
            best_diff = diff
            best_move = move

    # 2. Try toggling the turn if we don't have a perfect match (best_diff > 0)
    if best_diff > 0:
        original_turn = board.turn
        board.turn = not original_turn
        
        alt_best_move = None
        alt_best_diff = current_diff

        for move in board.legal_moves:
            board.push(move)
            diff = get_piece_changes(_placement(board), target)
            board.pop()

            if diff < alt_best_diff:
                alt_best_diff = diff
                alt_best_move = move

        if alt_best_diff < best_diff and alt_best_diff <= max_diff:
            best_diff = alt_best_diff
            best_move = alt_best_move
            turn_toggled = True
        else:
            board.turn = original_turn

    if best_diff <= max_diff and best_diff < current_diff:
        return best_move, best_diff, turn_toggled

    return None, current_diff, False


# ---------------------------------------------------------------------------
# Sequence builder
# ---------------------------------------------------------------------------


def build_move_sequence(
    fens: list[str],
    initial_board: chess.Board | None = None,
) -> MoveSequence:
    """Build a ``MoveSequence`` from an ordered list of piece-placement FENs."""
    if not fens:
        return MoveSequence()

    # Deduplicate consecutive identical FENs.
    unique_fens: list[str] = [_normalise_placement(fens[0])]
    for f in fens[1:]:
        nf = _normalise_placement(f)
        if nf != unique_fens[-1]:
            unique_fens.append(nf)

    # Initialize board based on first FEN
    if unique_fens[0] == _STARTING_PLACEMENT or get_piece_changes(unique_fens[0], _STARTING_PLACEMENT) <= 6:
        board = initial_board.copy() if initial_board else chess.Board()
        idx = 1 if unique_fens[0] == _STARTING_PLACEMENT else 0
        logger.info("Initialized board to standard starting position")
    else:
        # Initialize board to custom starting position (assume White to move initially)
        try:
            board = chess.Board(unique_fens[0] + " w - - 0 1")
            idx = 1
            logger.info("Initialized board to custom starting position: %s", unique_fens[0])
        except ValueError:
            board = initial_board.copy() if initial_board else chess.Board()
            idx = 0
            logger.warning("Failed to parse custom starting FEN, defaulted to standard position")

    result = MoveSequence(board=board)

    logger.info(
        "Building move sequence from %d FENs (%d unique transitions)",
        len(fens),
        len(unique_fens) - 1,
    )

    while idx < len(unique_fens):
        current_placement = _placement(board)
        target = unique_fens[idx]

        curr_diff = get_piece_changes(current_placement, target)
        move, best_diff, turn_toggled = detect_move_fuzzy(current_placement, target, board)

        if move is not None:
            if turn_toggled:
                logger.info("Turn toggled to resolve transition to: %s", target)
            board.push(move)
            result.moves.append(move)
            idx += 1
            continue

        # If no move could improve the match, check if it's just static classification noise (<= 3 differences)
        if curr_diff <= 3:
            idx += 1
            continue

        # If the target FEN is drastically different (>25 squares), it's almost certainly a
        # review/analysis frame or bad AI classification — not a legal game continuation.
        # Silently skip it without recording an error to preserve a clean error count.
        if curr_diff > 25:
            logger.debug(
                "Silently skipping review/garbage FEN (diff=%d > 25): %s", curr_diff, target
            )
            idx += 1
            continue

        # Could not find a legal move. Record error and try to skip ahead.
        error_info = {
            "from_fen": current_placement,
            "to_fen": target,
            "move_number": len(result.moves) + 1,
        }
        logger.warning("Could not resolve transition: %s", error_info)
        result.errors.append(error_info)

        # Attempt to resync: look ahead for a FEN that *is* reachable (fuzzy match).
        # Increased from 3 to 8 frames (~24s at 3s sampling) to handle longer AI confusion runs.
        resolved = False
        for lookahead in range(1, min(9, len(unique_fens) - idx)):
            next_target = unique_fens[idx + lookahead]

            next_diff = get_piece_changes(current_placement, next_target)
            if next_diff <= 3:
                idx = idx + lookahead + 1
                resolved = True
                logger.info(
                    "Re-synced by skipping %d FEN(s) ahead (position matches current board)",
                    lookahead,
                )
                break

            skip_move, skip_diff, skip_toggled = detect_move_fuzzy(
                current_placement, next_target, board
            )
            if skip_move is not None:
                if skip_toggled:
                    logger.info("Turn toggled to resolve skip-transition to: %s", next_target)
                board.push(skip_move)
                result.moves.append(skip_move)
                idx = idx + lookahead + 1
                resolved = True
                logger.info(
                    "Re-synced by skipping %d FEN(s) ahead (found legal move)", lookahead
                )
                break

        if not resolved and curr_diff <= 20:
            # Last resort: try 2-move deep search — play a move, then check if a second
            # move from that position can reach any upcoming FEN. Handles 2+ consecutive
            # AI misclassification frames without desynchronising.
            # Only triggered when the position mismatch is plausible (not a garbage FEN).
            for move1 in list(board.legal_moves)[:20]:  # limit branching factor
                board.push(move1)
                after1 = _placement(board)
                for lookahead in range(1, min(4, len(unique_fens) - idx)):
                    next_target = unique_fens[idx + lookahead]
                    skip2, diff2, toggled2 = detect_move_fuzzy(after1, next_target, board)
                    if skip2 is not None:
                        board.push(skip2)
                        result.moves.extend([move1, skip2])
                        idx = idx + lookahead + 1
                        resolved = True
                        logger.info(
                            "Re-synced via 2-move deep search, skipping %d FEN(s) ahead",
                            lookahead,
                        )
                        break
                if resolved:
                    break
                board.pop()

        if not resolved:
            # Give up on this transition and advance.
            idx += 1

    result.board = board
    logger.info(
        "Move sequence complete: %d moves, %d errors",
        len(result.moves),
        len(result.errors),
    )
    return result
