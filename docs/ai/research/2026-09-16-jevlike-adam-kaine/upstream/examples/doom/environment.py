"""ViZDoom environment and seven-button controller shared by the trainers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import vizdoom as vzd
from PIL import Image, ImageDraw


TICS_PER_ACTION = 4
TICS_PER_SECOND = 35
DOOM_ACTIONS = (
    "turn left",
    "turn right",
    "move forward",
    "move backward",
    "strafe left",
    "strafe right",
    "attack",
)
ACTION_BUTTONS = {
    "turn left": vzd.Button.TURN_LEFT,
    "turn right": vzd.Button.TURN_RIGHT,
    "move forward": vzd.Button.MOVE_FORWARD,
    "move backward": vzd.Button.MOVE_BACKWARD,
    "strafe left": vzd.Button.MOVE_LEFT,
    "strafe right": vzd.Button.MOVE_RIGHT,
    "attack": vzd.Button.ATTACK,
}
# Compatibility name used by the experiment scripts and released checkpoints.
DEFEND_ACTIONS = DOOM_ACTIONS


def make_game(seed: int = 7, scenario: str = "deadly_corridor",
              screen_resolution: str = "160x120") -> vzd.DoomGame:
    """Create a headless game with the same seven buttons in both supported levels."""
    game = vzd.DoomGame()
    game.load_config(str(Path(vzd.scenarios_path) / f"{scenario}.cfg"))
    if scenario == "defend_the_center":
        for button in (vzd.Button.MOVE_FORWARD, vzd.Button.MOVE_BACKWARD,
                       vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT):
            game.add_available_button(button)
    if scenario in ("defend_the_center", "deadly_corridor"):
        game.add_available_game_variable(vzd.GameVariable.KILLCOUNT)
        game.add_available_game_variable(vzd.GameVariable.HITCOUNT)
    if scenario == "deadly_corridor":
        game.add_available_game_variable(vzd.GameVariable.ARMOR)
    game.set_window_visible(False)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution({
        "160x120": vzd.ScreenResolution.RES_160X120,
        "640x480": vzd.ScreenResolution.RES_640X480,
    }[screen_resolution])
    game.set_labels_buffer_enabled(True)
    game.set_seed(seed)
    game.init()
    if len(game.get_available_buttons()) != len(DOOM_ACTIONS):
        raise RuntimeError(f"expected seven buttons, got {game.get_available_buttons()}")
    return game


def action_vector_for_game(index: int, names: tuple[str, ...],
                           game: vzd.DoomGame) -> list[bool]:
    selected = ACTION_BUTTONS[names[index]]
    return [button == selected for button in game.get_available_buttons()]


def draw_curve(values: list[float], path: Path) -> None:
    """Write a dependency-light training curve."""
    width, height, margin = 800, 420, 45
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.line((margin, 10, margin, height - margin, width - 10, height - margin),
              fill="black", width=2)
    if len(values) > 1:
        low, high = min(values), max(values)
        span = max(1.0, high - low)
        points = [
            (margin + i * (width - margin - 15) / (len(values) - 1),
             10 + (high - value) * (height - margin - 20) / span)
            for i, value in enumerate(values)
        ]
        draw.line(points, fill=(25, 100, 190), width=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
