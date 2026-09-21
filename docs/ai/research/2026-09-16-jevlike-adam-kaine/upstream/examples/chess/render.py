"""Draw a chess position as the screen the model sees.

The model never gets a side channel: the board, the cursor, and the "holding"
state are all pixels. The board is a square drawn inside the Doom frame size
(120x160) with black margins; the same drawing scales up for film frames.
"""

from __future__ import annotations

from functools import lru_cache

import chess
import numpy as np
from PIL import Image, ImageDraw

FRAME_HEIGHT, FRAME_WIDTH = 120, 160
BOARD = 120  # board side in the model frame -> 15 px squares

LIGHT, DARK = (201, 180, 140), (111, 81, 54)
WHITE_FILL, WHITE_LINE = (246, 241, 231), (28, 26, 23)
BLACK_FILL, BLACK_LINE = (28, 26, 23), (246, 241, 231)
CURSOR, CURSOR_HOLDING, ORIGIN = (255, 194, 71), (255, 90, 54), (224, 69, 60)
LEGAL = (70, 150, 230)


def screen_rc(square: int, flip: bool) -> tuple[int, int]:
    """Screen row/col (row 0 at top) of a chess square; flip puts black at the bottom."""
    file, rank = chess.square_file(square), chess.square_rank(square)
    return (rank, 7 - file) if flip else (7 - rank, file)


def square_at(row: int, col: int, flip: bool) -> int:
    return chess.square(7 - col, row) if flip else chess.square(col, 7 - row)


def _glyph(draw: ImageDraw.ImageDraw, piece: chess.Piece, s: int, fill, line, width: int,
           outline_only: bool = False) -> None:
    c, r = s / 2, s * 0.34
    fill = None if outline_only else fill
    kind = piece.piece_type
    if kind == chess.PAWN:
        draw.ellipse((c - r * 0.6, c - r * 0.6, c + r * 0.6, c + r * 0.6), fill, line, width)
    elif kind == chess.ROOK:
        draw.rectangle((c - r * 0.75, c - r * 0.75, c + r * 0.75, c + r * 0.75), fill, line, width)
    elif kind == chess.KNIGHT:
        draw.polygon([(c - r, c + r), (c + r, c + r), (c + r, c - r)], fill, line, width)
    elif kind == chess.BISHOP:
        draw.polygon([(c, c - r), (c + r, c), (c, c + r), (c - r, c)], fill, line, width)
    elif kind == chess.QUEEN:
        draw.ellipse((c - r, c - r, c + r, c + r), fill, line, width)
        draw.ellipse((c - r * 0.3, c - r * 0.3, c + r * 0.3, c + r * 0.3), line, line, width)
    else:  # king: a cross
        w = r * 0.38
        draw.polygon([(c - w, c - r), (c + w, c - r), (c + w, c - w), (c + r, c - w),
                      (c + r, c + w), (c + w, c + w), (c + w, c + r), (c - w, c + r),
                      (c - w, c + w), (c - r, c + w), (c - r, c - w), (c - w, c - w)],
                     fill, line, width)


@lru_cache(maxsize=8)
def _tiles(s: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per square side s: base tiles (2 colours x 13 pieces), ghost tiles, small sprites."""
    width = max(1, s // 15)
    base = np.zeros((2, 13, s, s, 3), np.uint8)
    ghost = np.zeros((2, 13, s, s, 3), np.uint8)
    small = np.zeros((13, s, s, 4), np.uint8)
    for colour, background in enumerate((LIGHT, DARK)):
        for code in range(13):
            image = Image.new("RGB", (s, s), background)
            piece = _piece(code)
            if piece:
                fill, line = (WHITE_FILL, WHITE_LINE) if piece.color else (BLACK_FILL, BLACK_LINE)
                _glyph(ImageDraw.Draw(image), piece, s, fill, line, width)
            base[colour, code] = np.asarray(image)
            image = Image.new("RGB", (s, s), background)
            if piece:
                _glyph(ImageDraw.Draw(image), piece, s, None, ORIGIN, width, outline_only=True)
            ghost[colour, code] = np.asarray(image)
    for code in range(1, 13):
        piece = _piece(code)
        fill, line = (WHITE_FILL, WHITE_LINE) if piece.color else (BLACK_FILL, BLACK_LINE)
        half = max(4, s // 2)
        image = Image.new("RGBA", (half, half), (0, 0, 0, 0))
        _glyph(ImageDraw.Draw(image), piece, half, fill, line, max(1, half // 15))
        small[code, :half, :half] = np.asarray(image)
    return base, ghost, small


def _piece(code: int) -> chess.Piece | None:
    if code == 0:
        return None
    return chess.Piece((code - 1) % 6 + 1, code <= 6)


def _code(piece: chess.Piece | None) -> int:
    return 0 if piece is None else piece.piece_type + (0 if piece.color else 6)


def render_board(board: chess.Board, cursor: tuple[int, int], flip: bool,
                 holding: int | None = None, size: int = BOARD) -> np.ndarray:
    """RGB (size, size, 3) picture of the position; cursor is (row, col) on screen."""
    s = size // 8
    base, ghost, small = _tiles(s)
    codes = np.zeros((8, 8), np.int64)
    for square, piece in board.piece_map().items():
        row, col = screen_rc(square, flip)
        codes[row, col] = _code(piece)
    colours = (np.add.outer(np.arange(8), np.arange(8)) % 2)
    held = 0
    if holding is not None:
        origin = screen_rc(holding, flip)
        held = codes[origin]
        codes[origin] = 0
    image = base[colours, codes]  # (8, 8, s, s, 3)
    if holding is not None:
        image = image.copy()
        image[origin] = ghost[colours[origin], held]
    image = image.transpose(0, 2, 1, 3, 4).reshape(8 * s, 8 * s, 3).copy()
    row, col = cursor
    y, x = row * s, col * s
    if holding is not None:
        _border(image, origin[0] * s, origin[1] * s, s, ORIGIN, max(1, s // 10))
        if codes[row, col] == 0:
            image[y:y + s, x:x + s] = base[colours[row, col], held]
        else:
            sprite = small[held]
            alpha = sprite[..., 3:4].astype(np.float32) / 255
            patch = image[y:y + s, x:x + s].astype(np.float32)
            image[y:y + s, x:x + s] = (patch * (1 - alpha) + sprite[..., :3] * alpha).astype(np.uint8)
    if holding is not None:  # legal destinations of the lifted piece, as chess sites show them
        d, lo = max(2, s // 5), (s - max(2, s // 5)) // 2
        for move in board.legal_moves:
            if move.from_square == holding:
                ty, tx = (v * s for v in screen_rc(move.to_square, flip))
                image[ty + lo:ty + lo + d, tx + lo:tx + lo + d] = LEGAL
    _border(image, y, x, s, CURSOR_HOLDING if holding is not None else CURSOR, max(1, s // 8))
    if size != 8 * s:
        image = np.asarray(Image.fromarray(image).resize((size, size), Image.NEAREST))
    return image


def _border(image: np.ndarray, y: int, x: int, s: int, colour, width: int) -> None:
    image[y:y + width, x:x + s] = colour
    image[y + s - width:y + s, x:x + s] = colour
    image[y:y + s, x:x + width] = colour
    image[y:y + s, x + s - width:x + s] = colour


def render_frame(board: chess.Board, cursor: tuple[int, int], flip: bool,
                 holding: int | None = None) -> np.ndarray:
    """The 120x160 RGB screen: the board centred, black margins either side."""
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
    left = (FRAME_WIDTH - BOARD) // 2
    frame[:, left:left + BOARD] = render_board(board, cursor, flip, holding, BOARD)
    return frame



def observation(board: chess.Board, cursor: tuple[int, int], holding: int | None = None) -> np.ndarray:
    """The model input, (120, 160, 4) uint8 in the Doom layout: RGB plus a motion channel.

    The mover is always at the bottom. Chess frames carry no motion, so the fourth
    channel is the Doom encoding of a zero frame difference (128).
    """
    frame = render_frame(board, cursor, board.turn == chess.BLACK, holding)
    return np.concatenate((frame, np.full((*frame.shape[:2], 1), 128, np.uint8)), -1)

if __name__ == "__main__":
    import time
    board = chess.Board()
    board.push_san("e4")
    frame = render_frame(board, (6, 4), flip=True, holding=chess.E7)
    assert frame.shape == (120, 160, 3) and frame[:, :20].max() == 0
    big = render_board(board, (6, 4), flip=True, holding=chess.E7, size=480)
    assert big.shape == (480, 480, 3)
    start = time.perf_counter()
    for _ in range(200):
        render_frame(board, (3, 3), flip=False)
    print(f"{(time.perf_counter() - start) * 1000 / 200:.3f} ms per 120x160 frame")
    import os
    os.makedirs("runs", exist_ok=True)
    Image.fromarray(big).save("runs/render-check.png")
    print("ok: runs/render-check.png")
