#!/usr/bin/env python3
"""Generate the Akh-Medu TTF font with 67 PUA glyphs.

35 fixed glyphs (U+E000-U+E022) for predicates, types, provenance, and structure.
32 radical primitives (U+E100-U+E11F) for composing dynamic VSA sigils.

Requires: pip install fonttools

Usage:
    python scripts/build_font.py
    # Output: fonts/akh-medu.ttf
"""

import os
import sys

try:
    from fontTools.fontBuilder import FontBuilder
    from fontTools.pens.t2Pen import T2Pen
except ImportError:
    print("Error: fonttools not installed. Run: pip install fonttools", file=sys.stderr)
    sys.exit(1)

# Font metrics (monospace, 1000 UPM).
UPM = 1000
ASCENT = 800
DESCENT = -200
ADVANCE_WIDTH = 600  # Monospace cell width.

# Margin for glyphs inside the cell.
M = 50
W = ADVANCE_WIDTH - 2 * M  # Usable width: 500
H = ASCENT - M  # Usable height from baseline: 750


def triangle_up(pen, cx, cy, size):
    """Upward-pointing triangle (pyramid) centered at (cx, cy)."""
    half = size // 2
    pen.moveTo((cx, cy + half))
    pen.lineTo((cx + half, cy - half))
    pen.lineTo((cx - half, cy - half))
    pen.closePath()


def triangle_down(pen, cx, cy, size):
    """Downward-pointing triangle."""
    half = size // 2
    pen.moveTo((cx, cy - half))
    pen.lineTo((cx + half, cy + half))
    pen.lineTo((cx - half, cy + half))
    pen.closePath()


def diamond(pen, cx, cy, size):
    """Diamond shape."""
    half = size // 2
    pen.moveTo((cx, cy + half))
    pen.lineTo((cx + half, cy))
    pen.lineTo((cx, cy - half))
    pen.lineTo((cx - half, cy))
    pen.closePath()


def circle(pen, cx, cy, r):
    """Approximate circle using cubic Bezier curves."""
    k = 0.5522847498  # Magic number for circular arc approximation.
    kr = int(k * r)
    pen.moveTo((cx, cy + r))
    pen.curveTo((cx + kr, cy + r), (cx + r, cy + kr), (cx + r, cy))
    pen.curveTo((cx + r, cy - kr), (cx + kr, cy - r), (cx, cy - r))
    pen.curveTo((cx - kr, cy - r), (cx - r, cy - kr), (cx - r, cy))
    pen.curveTo((cx - r, cy + kr), (cx - kr, cy + r), (cx, cy + r))
    pen.closePath()


def arc_left(pen, cx, cy, size):
    """Left-opening arc (subset symbol ⊂)."""
    half = size // 2
    k = int(0.55 * half)
    pen.moveTo((cx + half, cy + half))
    pen.curveTo((cx + half - k, cy + half), (cx, cy + k), (cx, cy))
    pen.curveTo((cx, cy - k), (cx + half - k, cy - half), (cx + half, cy - half))
    pen.endPath()


def arc_right(pen, cx, cy, size):
    """Right-opening arc (superset symbol ⊃)."""
    half = size // 2
    k = int(0.55 * half)
    pen.moveTo((cx - half, cy + half))
    pen.curveTo((cx - half + k, cy + half), (cx, cy + k), (cx, cy))
    pen.curveTo((cx, cy - k), (cx - half + k, cy - half), (cx - half, cy - half))
    pen.endPath()


def arrow_right(pen, cx, cy, size):
    """Right-pointing arrow."""
    half = size // 2
    qtr = size // 4
    pen.moveTo((cx - half, cy - qtr))
    pen.lineTo((cx, cy - qtr))
    pen.lineTo((cx, cy - half))
    pen.lineTo((cx + half, cy))
    pen.lineTo((cx, cy + half))
    pen.lineTo((cx, cy + qtr))
    pen.lineTo((cx - half, cy + qtr))
    pen.closePath()


def arrow_up(pen, cx, cy, size):
    """Up arrow."""
    half = size // 2
    qtr = size // 4
    pen.moveTo((cx, cy + half))
    pen.lineTo((cx + half, cy))
    pen.lineTo((cx + qtr, cy))
    pen.lineTo((cx + qtr, cy - half))
    pen.lineTo((cx - qtr, cy - half))
    pen.lineTo((cx - qtr, cy))
    pen.lineTo((cx - half, cy))
    pen.closePath()


def arrow_down(pen, cx, cy, size):
    """Down arrow."""
    half = size // 2
    qtr = size // 4
    pen.moveTo((cx, cy - half))
    pen.lineTo((cx + half, cy))
    pen.lineTo((cx + qtr, cy))
    pen.lineTo((cx + qtr, cy + half))
    pen.lineTo((cx - qtr, cy + half))
    pen.lineTo((cx - qtr, cy))
    pen.lineTo((cx - half, cy))
    pen.closePath()


def box_rect(pen, x, y, w, h):
    """Simple rectangle."""
    pen.moveTo((x, y))
    pen.lineTo((x + w, y))
    pen.lineTo((x + w, y + h))
    pen.lineTo((x, y + h))
    pen.closePath()


def star_5(pen, cx, cy, outer_r, inner_r):
    """5-pointed star."""
    import math
    points = []
    for i in range(10):
        r = outer_r if i % 2 == 0 else inner_r
        angle = math.pi / 2 + i * math.pi / 5
        px = cx + int(r * math.cos(angle))
        py = cy + int(r * math.sin(angle))
        points.append((px, py))
    pen.moveTo(points[0])
    for p in points[1:]:
        pen.lineTo(p)
    pen.closePath()


def cross(pen, cx, cy, size):
    """Plus/cross shape."""
    t = size // 6  # Thickness
    half = size // 2
    # Horizontal bar
    box_rect(pen, cx - half, cy - t, size, 2 * t)
    # Vertical bar
    box_rect(pen, cx - t, cy - half, 2 * t, size)


def wave(pen, cx, cy, size):
    """Wavy line."""
    half = size // 2
    qtr = size // 4
    pen.moveTo((cx - half, cy))
    pen.curveTo((cx - qtr, cy + qtr), (cx, cy + qtr), (cx, cy))
    pen.curveTo((cx, cy - qtr), (cx + qtr, cy - qtr), (cx + half, cy))
    pen.endPath()


def dot_glyph(pen, cx, cy, size):
    """Small filled dot."""
    circle(pen, cx, cy, size // 6)


def house(pen, cx, cy, size):
    """House: rectangle body + triangle roof."""
    half = size // 2
    qtr = size // 4
    # Body
    box_rect(pen, cx - half + qtr, cy - half, half, half)
    # Roof
    triangle_up(pen, cx, cy + qtr, half)


def pillar(pen, cx, cy, size):
    """Vertical pillar."""
    t = size // 8
    half = size // 2
    box_rect(pen, cx - t, cy - half, 2 * t, size)
    # Top cap
    box_rect(pen, cx - t * 2, cy + half - t, 4 * t, t)
    # Base
    box_rect(pen, cx - t * 2, cy - half, 4 * t, t)


def person(pen, cx, cy, size):
    """Simple person: circle head + triangle body."""
    head_r = size // 6
    circle(pen, cx, cy + size // 3, head_r)
    # Body triangle
    body_h = size // 2
    body_w = size // 3
    pen.moveTo((cx, cy + size // 3 - head_r - 2))
    pen.lineTo((cx + body_w, cy + size // 3 - head_r - body_h))
    pen.lineTo((cx - body_w, cy + size // 3 - head_r - body_h))
    pen.closePath()


# Define all 67 glyphs: (codepoint, name, draw_function).
CX = ADVANCE_WIDTH // 2
CY = (ASCENT + DESCENT) // 2
SZ = 400  # Standard glyph size


def draw_glyph(glyph_name, pen):
    """Draw a glyph by name into the given pen."""
    cx, cy, sz = CX, CY, SZ
    # -- Fixed glyphs (predicates) --
    if glyph_name == "is-a":
        triangle_up(pen, cx, cy, sz)
    elif glyph_name == "part-of":
        arc_left(pen, cx, cy, sz)
    elif glyph_name == "has-a":
        diamond(pen, cx, cy, sz)
    elif glyph_name == "contains":
        arc_right(pen, cx, cy, sz)
    elif glyph_name == "parent-of":
        arrow_down(pen, cx, cy, sz)
    elif glyph_name == "child-of":
        arrow_up(pen, cx, cy, sz)
    elif glyph_name == "similar-to":
        # Two parallel wavy lines
        wave(pen, cx, cy + 40, sz)
        wave(pen, cx, cy - 40, sz)
    elif glyph_name == "causes":
        arrow_right(pen, cx, cy, sz)
    elif glyph_name == "precedes":
        # Left angle bracket
        half = sz // 2
        pen.moveTo((cx + half // 2, cy + half))
        pen.lineTo((cx - half // 2, cy))
        pen.lineTo((cx + half // 2, cy - half))
        pen.endPath()
    elif glyph_name == "located-in":
        house(pen, cx, cy, sz)
    elif glyph_name == "created-by":
        # Simplified pencil: diagonal line + small triangle tip
        half = sz // 2
        pen.moveTo((cx - half, cy - half))
        pen.lineTo((cx + half, cy + half))
        pen.endPath()
        triangle_up(pen, cx + half - 30, cy + half - 30, 60)
    elif glyph_name == "depends-on":
        # Double left arrow
        arrow_right(pen, cx, cy, sz)  # We'll flip by drawing mirrored
    elif glyph_name == "opposes":
        # T-shape (perpendicular)
        t = sz // 8
        half = sz // 2
        box_rect(pen, cx - half, cy + half - t, sz, t)  # Top bar
        box_rect(pen, cx - t, cy - half, 2 * t, sz)  # Vertical
    elif glyph_name == "enables":
        arrow_right(pen, cx, cy, sz)
    elif glyph_name == "knows":
        # Bullseye: two concentric circles
        circle(pen, cx, cy, sz // 2)
        circle(pen, cx, cy, sz // 4)
    # -- Type determinatives --
    elif glyph_name == "type:person":
        person(pen, cx, cy, sz)
    elif glyph_name == "type:place":
        house(pen, cx, cy, sz)
    elif glyph_name == "type:thing":
        box_rect(pen, cx - sz // 3, cy - sz // 3, sz * 2 // 3, sz * 2 // 3)
    elif glyph_name == "type:concept":
        # Lightbulb: circle + small rectangle base
        circle(pen, cx, cy + 40, sz // 3)
        box_rect(pen, cx - 30, cy - sz // 4, 60, 40)
    elif glyph_name == "type:event":
        # Lightning bolt
        half = sz // 2
        pen.moveTo((cx, cy + half))
        pen.lineTo((cx + half // 3, cy))
        pen.lineTo((cx - half // 6, cy))
        pen.lineTo((cx, cy - half))
        pen.lineTo((cx - half // 3, cy))
        pen.lineTo((cx + half // 6, cy))
        pen.closePath()
    elif glyph_name == "type:quantity":
        # Hash symbol
        t = sz // 10
        third = sz // 3
        box_rect(pen, cx - third, cy - sz // 2, t, sz)
        box_rect(pen, cx + third - t, cy - sz // 2, t, sz)
        box_rect(pen, cx - sz // 2, cy - third + t, sz, t)
        box_rect(pen, cx - sz // 2, cy + third - t, sz, t)
    elif glyph_name == "type:time":
        # Clock: circle + two lines from center
        circle(pen, cx, cy, sz // 2)
        pen.moveTo((cx, cy))
        pen.lineTo((cx, cy + sz // 3))
        pen.endPath()
        pen.moveTo((cx, cy))
        pen.lineTo((cx + sz // 4, cy))
        pen.endPath()
    elif glyph_name == "type:group":
        # Three circles (people)
        r = sz // 6
        circle(pen, cx - r * 2, cy, r)
        circle(pen, cx, cy, r)
        circle(pen, cx + r * 2, cy, r)
    elif glyph_name == "type:process":
        # Gear: circle with notches
        circle(pen, cx, cy, sz // 3)
        # Simplified: just the circle for now
    elif glyph_name == "type:property":
        # Three horizontal lines (hamburger / triple bar)
        t = sz // 10
        for dy in [-sz // 4, 0, sz // 4]:
            box_rect(pen, cx - sz // 3, cy + dy - t // 2, sz * 2 // 3, t)
    # -- Provenance --
    elif glyph_name == "prov:asserted":
        diamond(pen, cx, cy, sz)  # Filled diamond
    elif glyph_name == "prov:inferred":
        diamond(pen, cx, cy, sz)  # Open diamond (same outline)
    elif glyph_name == "prov:fused":
        circle(pen, cx, cy, sz // 2)
        cross(pen, cx, cy, sz // 2)
    elif glyph_name == "prov:discovered":
        star_5(pen, cx, cy, sz // 2, sz // 4)
    elif glyph_name == "prov:gap":
        # Question mark: arc + dot
        circle(pen, cx, cy + sz // 4, sz // 4)
        dot_glyph(pen, cx, cy - sz // 3, sz)
    # -- Structural --
    elif glyph_name == "struct:triple":
        # Left angle bracket ⟨
        half = sz // 2
        pen.moveTo((cx + half // 2, cy + half))
        pen.lineTo((cx - half // 2, cy))
        pen.lineTo((cx + half // 2, cy - half))
        pen.endPath()
    elif glyph_name == "struct:end":
        # Right angle bracket ⟩
        half = sz // 2
        pen.moveTo((cx - half // 2, cy + half))
        pen.lineTo((cx + half // 2, cy))
        pen.lineTo((cx - half // 2, cy - half))
        pen.endPath()
    elif glyph_name == "struct:chain":
        # Horizontal line
        t = sz // 10
        box_rect(pen, cx - sz // 2, cy - t // 2, sz, t)
    elif glyph_name == "struct:branch":
        # T-junction (├)
        t = sz // 10
        half = sz // 2
        box_rect(pen, cx - t, cy - half, 2 * t, sz)  # Vertical
        box_rect(pen, cx, cy - t // 2, half, t)  # Right branch
    elif glyph_name == "struct:confidence":
        circle(pen, cx, cy, sz // 3)  # Filled circle
    # -- Radicals (beings) --
    elif glyph_name == "rad:eye":
        # Almond-shaped eye
        half = sz // 2
        k = half // 2
        pen.moveTo((cx - half, cy))
        pen.curveTo((cx - k, cy + k), (cx + k, cy + k), (cx + half, cy))
        pen.curveTo((cx + k, cy - k), (cx - k, cy - k), (cx - half, cy))
        pen.closePath()
        circle(pen, cx, cy, sz // 8)  # Pupil
    elif glyph_name == "rad:bird":
        # Simplified bird in flight: two arcs
        half = sz // 2
        pen.moveTo((cx - half, cy))
        pen.curveTo((cx - half // 2, cy + half), (cx, cy + half // 2), (cx, cy))
        pen.endPath()
        pen.moveTo((cx, cy))
        pen.curveTo((cx, cy + half // 2), (cx + half // 2, cy + half), (cx + half, cy))
        pen.endPath()
    elif glyph_name == "rad:serpent":
        wave(pen, cx, cy, sz)
    elif glyph_name == "rad:fish":
        # Fish: ellipse body + triangle tail
        circle(pen, cx - 20, cy, sz // 3)
        half = sz // 3
        pen.moveTo((cx + half, cy))
        pen.lineTo((cx + half + half // 2, cy + half // 2))
        pen.lineTo((cx + half + half // 2, cy - half // 2))
        pen.closePath()
    elif glyph_name == "rad:hand":
        # Simplified open hand: rectangle palm + 5 lines
        box_rect(pen, cx - sz // 4, cy - sz // 4, sz // 2, sz // 3)
        for i in range(5):
            x = cx - sz // 4 + i * (sz // 2) // 4
            pen.moveTo((x, cy + sz // 12))
            pen.lineTo((x, cy + sz // 3))
            pen.endPath()
    elif glyph_name == "rad:foot":
        # Foot: oval
        circle(pen, cx, cy - 30, sz // 3)
        box_rect(pen, cx - sz // 6, cy - sz // 3, sz // 3, sz // 4)
    elif glyph_name == "rad:face":
        circle(pen, cx, cy, sz // 2)
        # Eyes
        circle(pen, cx - sz // 6, cy + sz // 8, sz // 16)
        circle(pen, cx + sz // 6, cy + sz // 8, sz // 16)
    elif glyph_name == "rad:figure":
        person(pen, cx, cy, sz)
    # -- Radicals (nature) --
    elif glyph_name == "rad:sun":
        circle(pen, cx, cy, sz // 3)
        # Rays (simplified: small lines)
        import math
        for i in range(8):
            angle = i * math.pi / 4
            x1 = cx + int((sz // 3 + 20) * math.cos(angle))
            y1 = cy + int((sz // 3 + 20) * math.sin(angle))
            x2 = cx + int((sz // 2) * math.cos(angle))
            y2 = cy + int((sz // 2) * math.sin(angle))
            pen.moveTo((x1, y1))
            pen.lineTo((x2, y2))
            pen.endPath()
    elif glyph_name == "rad:moon":
        # Crescent: outer circle minus inner circle offset
        circle(pen, cx, cy, sz // 2)
    elif glyph_name == "rad:water":
        wave(pen, cx, cy + 40, sz)
        wave(pen, cx, cy, sz)
        wave(pen, cx, cy - 40, sz)
    elif glyph_name == "rad:mountain":
        triangle_up(pen, cx - sz // 4, cy, sz // 2)
        triangle_up(pen, cx + sz // 4, cy - sz // 8, sz * 3 // 4)
    elif glyph_name == "rad:tree":
        # Triangle crown + rectangle trunk
        triangle_up(pen, cx, cy + sz // 6, sz // 2)
        box_rect(pen, cx - sz // 16, cy - sz // 3, sz // 8, sz // 3)
    elif glyph_name == "rad:fire":
        # Flame: pointed oval
        triangle_up(pen, cx, cy, sz)
    elif glyph_name == "rad:wind":
        # Three curved lines
        for dy in [-40, 0, 40]:
            wave(pen, cx, cy + dy, sz)
    elif glyph_name == "rad:earth":
        # Circle with cross (earth symbol ⊕)
        circle(pen, cx, cy, sz // 2)
        cross(pen, cx, cy, sz)
    # -- Radicals (structure) --
    elif glyph_name == "rad:house":
        house(pen, cx, cy, sz)
    elif glyph_name == "rad:pillar":
        pillar(pen, cx, cy, sz)
    elif glyph_name == "rad:arch":
        # Arc / inverted U
        half = sz // 2
        k = int(0.55 * half)
        pen.moveTo((cx - half, cy - half))
        pen.lineTo((cx - half, cy))
        pen.curveTo((cx - half, cy + k), (cx - k, cy + half), (cx, cy + half))
        pen.curveTo((cx + k, cy + half), (cx + half, cy + k), (cx + half, cy))
        pen.lineTo((cx + half, cy - half))
        pen.endPath()
    elif glyph_name == "rad:wall":
        box_rect(pen, cx - sz // 2, cy - sz // 4, sz, sz // 2)
    elif glyph_name == "rad:gate":
        # Two pillars + arch
        t = sz // 8
        half = sz // 2
        box_rect(pen, cx - half, cy - half, t, sz)
        box_rect(pen, cx + half - t, cy - half, t, sz)
        box_rect(pen, cx - half, cy + half - t, sz, t)
    elif glyph_name == "rad:path":
        # Double horizontal line
        t = sz // 12
        box_rect(pen, cx - sz // 2, cy + t, sz, t)
        box_rect(pen, cx - sz // 2, cy - 2 * t, sz, t)
    elif glyph_name == "rad:bridge":
        # Arc with two supports
        half = sz // 2
        k = int(0.55 * half)
        pen.moveTo((cx - half, cy))
        pen.curveTo((cx - half, cy + k), (cx - k, cy + half // 2), (cx, cy + half // 2))
        pen.curveTo((cx + k, cy + half // 2), (cx + half, cy + k), (cx + half, cy))
        pen.endPath()
        t = sz // 10
        box_rect(pen, cx - half, cy - half, t, half)
        box_rect(pen, cx + half - t, cy - half, t, half)
    elif glyph_name == "rad:tower":
        circle(pen, cx, cy, sz // 2)  # Large circle
    # -- Radicals (abstract) --
    elif glyph_name == "rad:ankh":
        # Ankh: circle on top of T-shape
        circle(pen, cx, cy + sz // 4, sz // 5)
        t = sz // 10
        box_rect(pen, cx - t, cy - sz // 3, 2 * t, sz // 2)
        box_rect(pen, cx - sz // 4, cy, sz // 2, t)
    elif glyph_name == "rad:spiral":
        # Simplified spiral: circle with a line extending
        circle(pen, cx, cy, sz // 3)
        pen.moveTo((cx + sz // 3, cy))
        pen.curveTo((cx + sz // 2, cy + sz // 4), (cx + sz // 4, cy + sz // 2), (cx, cy + sz // 3))
        pen.endPath()
    elif glyph_name == "rad:star":
        star_5(pen, cx, cy, sz // 2, sz // 4)
    elif glyph_name == "rad:arrow":
        arrow_right(pen, cx, cy, sz)
    elif glyph_name == "rad:loop":
        # Infinity sign: two circles
        r = sz // 4
        circle(pen, cx - r, cy, r)
        circle(pen, cx + r, cy, r)
    elif glyph_name == "rad:cross":
        cross(pen, cx, cy, sz)
    elif glyph_name == "rad:wave":
        wave(pen, cx, cy, sz)
    elif glyph_name == "rad:dot":
        dot_glyph(pen, cx, cy, sz)
    else:
        # Fallback: small square
        box_rect(pen, cx - 20, cy - 20, 40, 40)


# Map codepoint -> glyph name for all 67 glyphs.
GLYPH_MAP = {
    # Fixed glyphs (35)
    0xE000: "is-a",        0xE001: "part-of",     0xE002: "has-a",
    0xE003: "contains",    0xE004: "parent-of",   0xE005: "child-of",
    0xE006: "similar-to",  0xE007: "causes",      0xE008: "precedes",
    0xE009: "located-in",  0xE00A: "created-by",  0xE00B: "depends-on",
    0xE00C: "opposes",     0xE00D: "enables",     0xE00E: "knows",
    0xE00F: "type:person", 0xE010: "type:place",  0xE011: "type:thing",
    0xE012: "type:concept",0xE013: "type:event",  0xE014: "type:quantity",
    0xE015: "type:time",   0xE016: "type:group",  0xE017: "type:process",
    0xE018: "type:property",
    0xE019: "prov:asserted",   0xE01A: "prov:inferred",
    0xE01B: "prov:fused",      0xE01C: "prov:discovered",
    0xE01D: "prov:gap",
    0xE01E: "struct:triple",   0xE01F: "struct:end",
    0xE020: "struct:chain",    0xE021: "struct:branch",
    0xE022: "struct:confidence",
    # Radicals (32)
    0xE100: "rad:eye",     0xE101: "rad:bird",    0xE102: "rad:serpent",
    0xE103: "rad:fish",    0xE104: "rad:hand",    0xE105: "rad:foot",
    0xE106: "rad:face",    0xE107: "rad:figure",
    0xE108: "rad:sun",     0xE109: "rad:moon",    0xE10A: "rad:water",
    0xE10B: "rad:mountain",0xE10C: "rad:tree",    0xE10D: "rad:fire",
    0xE10E: "rad:wind",    0xE10F: "rad:earth",
    0xE110: "rad:house",   0xE111: "rad:pillar",  0xE112: "rad:arch",
    0xE113: "rad:wall",    0xE114: "rad:gate",    0xE115: "rad:path",
    0xE116: "rad:bridge",  0xE117: "rad:tower",
    0xE118: "rad:ankh",    0xE119: "rad:spiral",  0xE11A: "rad:star",
    0xE11B: "rad:arrow",   0xE11C: "rad:loop",    0xE11D: "rad:cross",
    0xE11E: "rad:wave",    0xE11F: "rad:dot",
}


def build_font():
    """Build the Akh-Medu TTF font."""
    # Glyph names and codepoint-to-glyph mapping.
    glyph_names = [".notdef", "space"]
    cmap = {0x20: "space"}  # Space character
    char_strings = {}

    # .notdef glyph (empty).
    fb = FontBuilder(UPM, isTTF=False)  # CFF-based for cleaner outlines

    # Build all 67 glyphs.
    for codepoint, name in sorted(GLYPH_MAP.items()):
        glyph_name = f"uni{codepoint:04X}"
        glyph_names.append(glyph_name)
        cmap[codepoint] = glyph_name

    fb.setupGlyphOrder(glyph_names)

    # Draw glyphs.
    fb.setupCharacterMap(cmap)

    pen_dict = {}

    # .notdef: empty
    pen = fb.setupGlyf({}) if False else None  # CFF mode, we'll use T2 pens

    # For CFF, we need to draw into charstrings.
    glyph_dict = {}
    for glyph_name in glyph_names:
        pen = T2Pen(ADVANCE_WIDTH, None)
        if glyph_name == ".notdef":
            # Draw a small rectangle for .notdef
            box_rect(pen, 100, 0, 400, 700)
        elif glyph_name == "space":
            pass  # Empty
        else:
            # Extract codepoint from glyph name (uniXXXX).
            cp = int(glyph_name[3:], 16)
            draw_name = GLYPH_MAP.get(cp, "")
            draw_glyph(draw_name, pen)
        glyph_dict[glyph_name] = pen.getCharString()

    fb.setupCFF(
        nameStrings={"version": "1.0"},
        charStringsDict=glyph_dict,
        privateDict={"defaultWidthX": ADVANCE_WIDTH},
    )

    metrics = {}
    for glyph_name in glyph_names:
        metrics[glyph_name] = (ADVANCE_WIDTH, 0)  # (width, lsb)

    fb.setupHorizontalMetrics(metrics)

    fb.setupHorizontalHeader(ascent=ASCENT, descent=DESCENT)
    fb.setupNameTable({
        "familyName": "Akh-Medu",
        "styleName": "Regular",
    })
    fb.setupOs2(
        sTypoAscender=ASCENT,
        sTypoDescender=DESCENT,
        sTypoLineGap=0,
        usWinAscent=ASCENT,
        usWinDescent=abs(DESCENT),
        sxHeight=500,
        sCapHeight=700,
    )
    fb.setupPost()

    # Output.
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fonts")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "akh-medu.otf")
    fb.font.save(out_path)
    print(f"Font generated: {out_path}")
    print(f"  {len(GLYPH_MAP)} PUA glyphs (35 fixed + 32 radicals)")
    print(f"  UPM: {UPM}, Advance width: {ADVANCE_WIDTH}")
    print(f"\nTo install:")
    print(f"  cp {out_path} ~/.local/share/fonts/akh-medu.otf")
    print(f"  fc-cache -fv")


if __name__ == "__main__":
    build_font()
