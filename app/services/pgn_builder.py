"""PGN builder service.

Constructs standard PGN text from move sequences and game metadata using
the ``python-chess`` library.  Supports both single-game and multi-game
output and writes files that are directly importable into Chess.com and
Lichess.
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, Union

import chess
import chess.pgn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class GameMetadata:
    """Metadata headers for a PGN game."""

    event: str = "Video Analysis"
    site: str = "Chess.com"
    date: str = field(default_factory=lambda: date.today().strftime("%Y.%m.%d"))
    round_num: str = "?"
    white: str = "?"
    black: str = "?"
    result: str = "*"
    source_url: str = ""


class PGNBuildError(Exception):
    """Raised when PGN construction fails."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_game(
    moves: list[chess.Move],
    metadata: GameMetadata,
    starting_fen: Optional[str] = None,
) -> chess.pgn.Game:
    """Create a ``chess.pgn.Game`` from *moves* and *metadata*.

    Parameters
    ----------
    moves:
        Ordered list of moves.
    metadata:
        Game metadata for PGN headers.
    starting_fen:
        Optional FEN string for a custom starting position.  When provided
        the ``SetUp`` and ``FEN`` headers are added so that PGN consumers
        know the game does not start from the standard position.
    """
    game = chess.pgn.Game()

    # Set standard seven-tag roster headers.
    game.headers["Event"] = metadata.event
    game.headers["Site"] = metadata.site
    game.headers["Date"] = metadata.date
    game.headers["Round"] = metadata.round_num
    game.headers["White"] = metadata.white
    game.headers["Black"] = metadata.black
    game.headers["Result"] = metadata.result

    # Optional: source URL as a non-standard header.
    if metadata.source_url:
        game.headers["SourceURL"] = metadata.source_url

    # Custom starting position support.
    if starting_fen is not None:
        game.headers["SetUp"] = "1"
        game.headers["FEN"] = starting_fen
        board = chess.Board(starting_fen)
    else:
        board = game.board()

    # Replay moves onto the game node tree.
    node: chess.pgn.GameNode = game

    for move in moves:
        if move not in board.legal_moves:
            logger.warning(
                "Illegal move %s at position %s – stopping replay",
                move.uci(),
                board.fen(),
            )
            break
        node = node.add_variation(move)
        board.push(move)

    game.headers["Result"] = metadata.result
    return game


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_pgn(
    moves: list[chess.Move],
    metadata: GameMetadata | None = None,
    starting_fen: Optional[str] = None,
) -> str:
    """Build a PGN string for a single game.

    Parameters
    ----------
    moves:
        Ordered list of moves from the starting position.
    metadata:
        Optional game metadata; defaults are used when omitted.
    starting_fen:
        Optional FEN string for a custom starting position.

    Returns
    -------
    str
        A complete PGN string (headers + move text).
    """
    if metadata is None:
        metadata = GameMetadata()

    game = _create_game(moves, metadata, starting_fen=starting_fen)

    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
    pgn_text: str = game.accept(exporter)

    logger.info("Built PGN: %d moves, result=%s", len(moves), metadata.result)
    return pgn_text


def build_multi_game_pgn(
    games: list[
        Union[
            tuple[list[chess.Move], GameMetadata],
            tuple[list[chess.Move], GameMetadata, Optional[str]],
        ]
    ],
) -> str:
    """Build a PGN string containing multiple games.

    Parameters
    ----------
    games:
        List of game tuples.  Each element may be either
        ``(moves, metadata)`` or ``(moves, metadata, starting_fen)``.

    Returns
    -------
    str
        Combined PGN with games separated by blank lines.
    """
    parts: list[str] = []
    for idx, game_tuple in enumerate(games, start=1):
        if len(game_tuple) == 3:
            moves, meta, fen = game_tuple  # type: ignore[misc]
        else:
            moves, meta = game_tuple  # type: ignore[misc]
            fen = None
        if meta.round_num == "?":
            meta.round_num = str(idx)
        parts.append(build_pgn(moves, meta, starting_fen=fen))

    combined = "\n\n".join(parts) + "\n"
    logger.info("Built multi-game PGN: %d game(s)", len(games))
    return combined


def save_pgn(pgn_string: str, output_path: str) -> None:
    """Write a PGN string to a file.

    Parameters
    ----------
    pgn_string:
        The PGN content to write.
    output_path:
        Destination file path.  Parent directories are created if needed.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pgn_string)
    logger.info("Saved PGN to %s", output_path)
