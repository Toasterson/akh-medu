"""The chess controller: four arrows and one pick-up/put-down toggle.

Screen coordinates are (row, col) with row 0 at the top; the mover's pieces are
always at the bottom (the board is flipped when black is to move).
"""

from __future__ import annotations

import chess

from render import screen_rc, square_at

KEYS = ("move up", "move down", "move left", "move right", "pick up / put down")
UP, DOWN, LEFT, RIGHT, TOGGLE = range(5)
STEP = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}


def path_keys(cursor: tuple[int, int], target: tuple[int, int]) -> list[int]:
    """Fixed deterministic path: rows first, then columns."""
    keys = []
    row, col = cursor
    while row != target[0]:
        keys.append(UP if target[0] < row else DOWN)
        row += -1 if target[0] < row else 1
    while col != target[1]:
        keys.append(LEFT if target[1] < col else RIGHT)
        col += -1 if target[1] < col else 1
    return keys


def move_keys(cursor: tuple[int, int], move: chess.Move, flip: bool) -> list[int]:
    """Keys that play `move` from `cursor`: walk to the piece, lift, walk, put down."""
    origin, target = screen_rc(move.from_square, flip), screen_rc(move.to_square, flip)
    return path_keys(cursor, origin) + [TOGGLE] + path_keys(origin, target) + [TOGGLE]


class Controller:
    """Applies keys to a board. `press` returns what happened; wasted presses are named."""

    def __init__(self, board: chess.Board, cursor: tuple[int, int] = (7, 4)) -> None:
        self.board = board
        self.cursor = cursor
        self.holding: int | None = None

    @property
    def flip(self) -> bool:
        return self.board.turn == chess.BLACK

    def press(self, key: int) -> tuple[str, chess.Move | None]:
        if key != TOGGLE:
            dr, dc = STEP[key]
            row, col = self.cursor[0] + dr, self.cursor[1] + dc
            if not (0 <= row < 8 and 0 <= col < 8):
                return "edge", None
            self.cursor = (row, col)
            return "moved", None
        square = square_at(*self.cursor, self.flip)
        if self.holding is None:
            piece = self.board.piece_at(square)
            if piece is None or piece.color != self.board.turn:
                return "empty", None
            self.holding = square
            return "lift", None
        if square == self.holding:
            self.holding = None
            return "drop", None
        move = chess.Move(self.holding, square)
        if self.board.piece_type_at(self.holding) == chess.PAWN and chess.square_rank(square) in (0, 7):
            move = chess.Move(self.holding, square, promotion=chess.QUEEN)
        if move not in self.board.legal_moves:
            return "illegal", None
        self.board.push(move)
        self.holding = None
        return "put", move


WASTED = {"edge", "empty", "illegal"}


if __name__ == "__main__":
    board = chess.Board()
    controller = Controller(board, cursor=(0, 0))
    keys = move_keys(controller.cursor, chess.Move.from_uci("e2e4"), flip=False)
    assert keys == [DOWN] * 6 + [RIGHT] * 4 + [TOGGLE, UP, UP, TOGGLE], keys
    events = [controller.press(k)[0] for k in keys]
    assert events[-1] == "put" and board.fen().startswith("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b")
    # black to move: flipped board, e7e5 from the cursor left where it was (screen row 4, col 4 = e4)
    keys = move_keys(controller.cursor, chess.Move.from_uci("e7e5"), flip=True)
    assert controller.flip and keys == [DOWN, DOWN, LEFT, TOGGLE, UP, UP, TOGGLE], keys
    assert [controller.press(k)[0] for k in keys][-1] == "put"
    assert controller.press(UP)[0] == "moved" and controller.press(TOGGLE)[0] == "empty"
    controller.cursor = (7, 7)
    assert controller.press(DOWN)[0] == "edge"
    print("ok")
