#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import shutil
import subprocess
from pathlib import Path


WIDTH = 1500
HEIGHT = 730

BG = "#ffffff"
STROKE = "#222222"
MUTED = "#5a5a5a"
LIGHT = "#f6f6f6"
GRID = "#c9c9c9"

PANEL_Y = 45
PANEL_H = 620
PANEL_A_X = 60
PANEL_A_W = 660
PANEL_B_X = 780
PANEL_B_W = 660

A_ROWS_Y = [255, 335, 415, 495]
A_LEFT_X = PANEL_A_X + 88
A_BLOCK_X = PANEL_A_X + 272
A_BLOCK_Y = 200
A_BLOCK_W = 210
A_BLOCK_H = 335
A_RIGHT_X = PANEL_A_X + 565
A_RIGHT_LABEL_X = PANEL_A_X + 602

B_DIVIDER_X = PANEL_B_X + PANEL_B_W / 2
B_LEFT_CENTER_X = PANEL_B_X + PANEL_B_W / 4
B_RIGHT_CENTER_X = PANEL_B_X + (3 * PANEL_B_W) / 4
HAD_CELL = 54
HAD_GRID_X = B_LEFT_CENTER_X - 57
HAD_GRID_Y = 235
HAD_FACTOR_X = HAD_GRID_X - 34
HOUSE_X = B_RIGHT_CENTER_X


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    arrow: bool = False,
    stroke_width: float = 3.0,
    color: str = STROKE,
) -> str:
    marker = ' marker-end="url(#arrow)"' if arrow else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'fill="none" stroke="{color}" stroke-width="{stroke_width}" '
        f'stroke-linecap="round" stroke-linejoin="round"{marker} />'
    )


def rect(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str = BG,
    stroke: str = STROKE,
    stroke_width: float = 2.5,
    rx: float = 0,
) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" />'
    )


def text(
    x: float,
    y: float,
    content: str,
    *,
    size: int = 28,
    anchor: str = "middle",
    weight: str = "400",
    color: str = STROKE,
    italic: bool = False,
) -> str:
    font_style = "italic" if italic else "normal"
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}" '
        f'font-style="{font_style}">{html.escape(content)}</text>'
    )


def rich_text(
    x: float,
    y: float,
    spans: list[dict[str, object]],
    *,
    size: int = 28,
    anchor: str = "middle",
    weight: str = "400",
    color: str = STROKE,
) -> str:
    parts = [
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}" '
        'xml:space="preserve">'
    ]
    for span in spans:
        attrs = []
        if span.get("italic"):
            attrs.append('font-style="italic"')
        if "size" in span:
            attrs.append(f'font-size="{span["size"]}"')
        if "baseline_shift" in span:
            attrs.append(f'baseline-shift="{span["baseline_shift"]}"')
        if "dx" in span:
            attrs.append(f'dx="{span["dx"]}"')
        parts.append(
            f'<tspan {" ".join(attrs)}>{html.escape(str(span["text"]))}</tspan>'
        )
    parts.append("</text>")
    return "".join(parts)


def multiline_text(
    x: float,
    y: float,
    lines: list[str],
    *,
    size: int = 24,
    line_gap: int = 32,
    anchor: str = "middle",
    weight: str = "400",
    color: str = STROKE,
) -> str:
    parts = [
        (
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
            f'font-size="{size}" font-weight="{weight}" fill="{color}">'
        )
    ]
    for idx, content in enumerate(lines):
        dy = "0" if idx == 0 else str(line_gap)
        parts.append(f'<tspan x="{x}" dy="{dy}">{html.escape(content)}</tspan>')
    parts.append("</text>")
    return "".join(parts)


def indexed_symbol(
    x: float,
    y: float,
    symbol: str,
    index: int,
    *,
    anchor: str = "middle",
    size: int = 24,
    weight: str = "400",
    color: str = STROKE,
) -> str:
    return rich_text(
        x,
        y,
        [
            {"text": symbol, "italic": True},
            {"text": str(index), "size": int(size * 0.68), "baseline_shift": "sub"},
        ],
        size=size,
        anchor=anchor,
        weight=weight,
        color=color,
    )


def hadamard_grid() -> str:
    signs = [
        ["+", "+", "+", "+"],
        ["+", "-", "+", "-"],
        ["+", "+", "-", "-"],
        ["+", "-", "-", "+"],
    ]
    parts: list[str] = []
    parts.append(text(HAD_FACTOR_X, HAD_GRID_Y + 2 * HAD_CELL, "1/2 x", size=26, anchor="end", weight="500"))
    parts.append(rect(HAD_GRID_X, HAD_GRID_Y, 4 * HAD_CELL, 4 * HAD_CELL, fill=BG, stroke=STROKE, stroke_width=2.2))
    for row in range(4):
        for col in range(4):
            x = HAD_GRID_X + col * HAD_CELL
            y = HAD_GRID_Y + row * HAD_CELL
            parts.append(rect(x, y, HAD_CELL, HAD_CELL, fill=LIGHT, stroke=GRID, stroke_width=1.2))
            parts.append(text(x + HAD_CELL / 2, y + HAD_CELL / 2 + 1, signs[row][col], size=28, weight="500"))
    parts.append(rect(HAD_GRID_X, HAD_GRID_Y, 4 * HAD_CELL, 4 * HAD_CELL, fill="none", stroke=STROKE, stroke_width=2.2))
    return "\n".join(parts)


def panel_a() -> str:
    parts: list[str] = []
    parts.append(rect(PANEL_A_X, PANEL_Y, PANEL_A_W, PANEL_H, fill=BG, stroke=STROKE, stroke_width=2.2))
    parts.append(text(PANEL_A_X + 36, PANEL_Y + 38, "(a)", size=24, weight="600"))
    parts.append(text(PANEL_A_X + PANEL_A_W / 2, PANEL_Y + 38, "Feedback matrix as mixing stage", size=28, weight="500"))

    parts.append(text(PANEL_A_X + 178, PANEL_Y + 92, "Delay-line outputs", size=22, weight="500"))
    parts.append(text(PANEL_A_X + 563, PANEL_Y + 92, "Mixed inputs", size=22, weight="500"))

    parts.append(rect(A_BLOCK_X, A_BLOCK_Y, A_BLOCK_W, A_BLOCK_H, fill=LIGHT, stroke=STROKE, stroke_width=2.6))

    mix_map = [2, 0, 3, 1]
    for idx, row_y in enumerate(A_ROWS_Y):
        out_y = A_ROWS_Y[mix_map[idx]]
        parts.append(
            line(
                A_BLOCK_X + 22,
                row_y,
                A_BLOCK_X + A_BLOCK_W - 22,
                out_y,
                stroke_width=2.3,
                color=MUTED,
            )
        )

    parts.append(multiline_text(A_BLOCK_X + A_BLOCK_W / 2, PANEL_Y + 224, ["Feedback", "matrix A"], size=26, line_gap=30, weight="600"))

    for idx, row_y in enumerate(A_ROWS_Y, start=1):
        parts.append(indexed_symbol(A_LEFT_X - 18, row_y, "d", idx, size=24, anchor="end"))
        parts.append(line(A_LEFT_X, row_y, A_BLOCK_X, row_y, arrow=True, stroke_width=3.0))
        parts.append(line(A_BLOCK_X + A_BLOCK_W, row_y, A_RIGHT_X, row_y, arrow=True, stroke_width=3.0))
        parts.append(indexed_symbol(A_RIGHT_LABEL_X, row_y, "u", idx, size=22, anchor="start", color=MUTED))
    return "\n".join(parts)


def panel_b() -> str:
    parts: list[str] = []
    parts.append(rect(PANEL_B_X, PANEL_Y, PANEL_B_W, PANEL_H, fill=BG, stroke=STROKE, stroke_width=2.2))
    parts.append(text(PANEL_B_X + 36, PANEL_Y + 38, "(b)", size=24, weight="600"))
    parts.append(text(PANEL_B_X + PANEL_B_W / 2, PANEL_Y + 38, "Examples of unilossless matrix families", size=28, weight="500"))
    parts.append(line(B_DIVIDER_X, PANEL_Y + 82, B_DIVIDER_X, PANEL_Y + PANEL_H - 42, stroke_width=1.6, color=GRID))

    parts.append(text(B_LEFT_CENTER_X, PANEL_Y + 132, "Normalized Hadamard", size=25, weight="500"))
    parts.append(hadamard_grid())
    parts.append(multiline_text(B_LEFT_CENTER_X, PANEL_Y + 545, ["Fixed structured", "orthogonal mixer"], size=20, line_gap=24, color=MUTED, weight="500"))

    parts.append(text(HOUSE_X, PANEL_Y + 132, "Householder form", size=25, weight="500"))
    parts.append(
        rich_text(
            HOUSE_X,
            PANEL_Y + 286,
            [
                {"text": "H", "italic": True},
                {"text": " = I - 2 "},
                {"text": "u", "italic": True},
                {"text": " "},
                {"text": "u", "italic": True},
                {"text": "T", "size": 24, "baseline_shift": "super"},
            ],
            size=34,
            weight="500",
        )
    )
    parts.append(
        multiline_text(
            HOUSE_X,
            PANEL_Y + 545,
            ["Compact parameterized", "orthogonal mixer"],
            size=20,
            line_gap=26,
            color=MUTED,
            weight="500",
        )
    )
    return "\n".join(parts)


def svg_markup() -> str:
    parts: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" '
            'role="img" aria-labelledby="title desc">'
        ),
        "<title id=\"title\">Unilossless feedback mixing in a 4x4 FDN</title>",
        (
            "<desc id=\"desc\">Two-panel academic-style figure showing the feedback "
            "matrix as a mixing stage in a 4x4 feedback delay network and "
            "illustrating Hadamard and Householder unilossless matrix families.</desc>"
        ),
        "<defs>",
        (
            f'<marker id="arrow" markerWidth="11" markerHeight="11" refX="9.5" '
            f'refY="5.5" orient="auto" markerUnits="strokeWidth">'
            f'<path d="M 0 0 L 11 5.5 L 0 11 z" fill="{STROKE}" /></marker>'
        ),
        "<style><![CDATA[",
        f"svg {{ background: {BG}; }}",
        "text { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; dominant-baseline: middle; }",
        "]]></style>",
        "</defs>",
        f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="{BG}" />',
        panel_a(),
        panel_b(),
        "</svg>",
    ]
    return "\n".join(parts)


def convert_with_rsvg(svg_path: Path, output_path: Path, fmt: str) -> bool:
    rsvg = shutil.which("rsvg-convert")
    if not rsvg:
        return False
    subprocess.run([rsvg, "-f", fmt, "-o", str(output_path), str(svg_path)], check=True)
    return True


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="Generate the Section 2.3 unilossless feedback mixing figure."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "paper_assets" / "figures",
        help="Directory for rendered figure outputs.",
    )
    parser.add_argument(
        "--basename",
        default="fig_unilossless_feedback_mixing_4x4",
        help="Base filename for exported assets without extension.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    svg_path = out_dir / f"{args.basename}.svg"
    png_path = out_dir / f"{args.basename}.png"
    pdf_path = out_dir / f"{args.basename}.pdf"

    svg_path.write_text(svg_markup(), encoding="utf-8")
    print(f"Wrote {svg_path}")

    for fmt, path in [("png", png_path), ("pdf", pdf_path)]:
        try:
            if convert_with_rsvg(svg_path, path, fmt):
                print(f"Wrote {path}")
            else:
                print(f"Skipped {path} (rsvg-convert not found)")
        except subprocess.CalledProcessError as exc:
            print(f"Skipped {path} ({exc})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
