#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import shutil
import subprocess
from pathlib import Path


WIDTH = 1400
HEIGHT = 760

BG = "#ffffff"
STROKE = "#222222"
MUTED = "#4a4a4a"
BLOCK_FILL = "#ffffff"

ROWS_Y = [220, 330, 440, 550]
JUNCTION_X = [320, 340, 360, 380]
INPUT_BUS_X = 150
SUM_R = 13
INPUT_LABEL_X = 210
INPUT_LABEL_Y = 174

DELAY_X = 430
DELAY_Y = [y - 30 for y in ROWS_Y]
DELAY_W = 240
DELAY_H = 60
DELAY_OUT_X = DELAY_X + DELAY_W

MATRIX_X = 250
MATRIX_Y = 60
MATRIX_W = 650
MATRIX_H = 95
MATRIX_BOTTOM_Y = MATRIX_Y + MATRIX_H

BRANCH_X = [720, 770, 820, 870]

MIX_X = 1000
MIX_Y = 180
MIX_W = 190
MIX_H = 410
OUTPUT_Y = MIX_Y + MIX_H / 2
OUTPUT_END_X = 1290


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
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"'
        f' fill="none" stroke="{color}" stroke-width="{stroke_width}"'
        f' stroke-linecap="round" stroke-linejoin="round"{marker} />'
    )


def rect(x: float, y: float, w: float, h: float) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
        f'fill="{BLOCK_FILL}" stroke="{STROKE}" stroke-width="3" />'
    )


def dot(x: float, y: float, *, r: float = 4.5) -> str:
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{STROKE}" />'


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


def multiline_text(
    x: float,
    y: float,
    lines: list[str],
    *,
    size: int = 28,
    line_gap: int = 34,
    anchor: str = "middle",
    weight: str = "400",
    italic_last: bool = False,
) -> str:
    parts = [
        (
            f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
            f'font-size="{size}" font-weight="{weight}" fill="{STROKE}">'
        )
    ]
    for idx, content in enumerate(lines):
        dy = "0" if idx == 0 else str(line_gap)
        font_style = "italic" if italic_last and idx == len(lines) - 1 else "normal"
        parts.append(
            f'<tspan x="{x}" dy="{dy}" font-style="{font_style}">'
            f"{html.escape(content)}</tspan>"
        )
    parts.append("</text>")
    return "".join(parts)


def sum_circle(x: float, y: float) -> str:
    return (
        f'<circle cx="{x}" cy="{y}" r="{SUM_R}" fill="{BG}" '
        f'stroke="{STROKE}" stroke-width="2.5" />'
        + text(x, y + 1, "+", size=20, weight="500")
    )


def svg_markup() -> str:
    parts: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" '
            f'role="img" aria-labelledby="title desc">'
        ),
        "<title id=\"title\">4x4 FDN block diagram</title>",
        (
            "<desc id=\"desc\">Simplified academic-style signal-flow diagram of a "
            "4x4 feedback delay network with mono input, four delay lines, "
            "feedback matrix, output mix, and mono output.</desc>"
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
    ]

    parts.append(rect(MATRIX_X, MATRIX_Y, MATRIX_W, MATRIX_H))
    parts.append(
        text(
            MATRIX_X + MATRIX_W / 2,
            MATRIX_Y + MATRIX_H / 2 + 1,
            "Feedback matrix A",
            size=30,
            weight="500",
        )
    )

    parts.append(line(92, OUTPUT_Y, INPUT_BUS_X, OUTPUT_Y, arrow=True))
    parts.append(
        text(74, OUTPUT_Y, "x[n]", size=30, anchor="end", italic=True)
    )
    parts.append(
        rich_text(
            INPUT_LABEL_X,
            INPUT_LABEL_Y,
            [
                {"text": "Injection vector"},
                {"text": "b", "italic": True, "dx": "4"},
            ],
            size=22,
            anchor="middle",
            weight="500",
        )
    )
    parts.append(
        line(INPUT_BUS_X, ROWS_Y[0], INPUT_BUS_X, ROWS_Y[-1], stroke_width=2.6, color=MUTED)
    )

    parts.append(rect(MIX_X, MIX_Y, MIX_W, MIX_H))
    parts.append(
        text(
            MIX_X + MIX_W / 2,
            OUTPUT_Y - 22,
            "Output mix",
            size=28,
            weight="500",
        )
    )
    parts.append(
        rich_text(
            MIX_X + MIX_W / 2,
            OUTPUT_Y + 20,
            [
                {"text": "c", "italic": True},
                {"text": "T", "size": 18, "baseline_shift": "super"},
            ],
            size=27,
            weight="500",
        )
    )
    parts.append(line(MIX_X + MIX_W, OUTPUT_Y, OUTPUT_END_X, OUTPUT_Y, arrow=True))
    parts.append(
        text(OUTPUT_END_X + 24, OUTPUT_Y, "y[n]", size=30, anchor="start", italic=True)
    )

    for idx, row_y in enumerate(ROWS_Y, start=1):
        jx = JUNCTION_X[idx - 1]
        dy = DELAY_Y[idx - 1]
        bx = BRANCH_X[idx - 1]

        parts.append(line(INPUT_BUS_X, row_y, jx - SUM_R, row_y, arrow=True))
        parts.append(line(jx, MATRIX_BOTTOM_Y, jx, row_y - SUM_R, arrow=True))
        parts.append(sum_circle(jx, row_y))

        parts.append(line(jx + SUM_R, row_y, DELAY_X, row_y, arrow=True))
        parts.append(rect(DELAY_X, dy, DELAY_W, DELAY_H))
        parts.append(
            text(
                DELAY_X + DELAY_W / 2,
                row_y + 1,
                f"Delay m{idx}",
                size=28,
                weight="500",
            )
        )

        parts.append(line(DELAY_OUT_X, row_y, bx, row_y))
        parts.append(dot(bx, row_y, r=4.2))
        parts.append(line(bx, row_y, bx, MATRIX_BOTTOM_Y, arrow=True))
        parts.append(line(bx, row_y, MIX_X, row_y, arrow=True))

    parts.append("</svg>")
    return "\n".join(parts)


def convert_with_rsvg(svg_path: Path, output_path: Path, fmt: str) -> bool:
    rsvg = shutil.which("rsvg-convert")
    if not rsvg:
        return False
    command = [rsvg, "-f", fmt, "-o", str(output_path), str(svg_path)]
    subprocess.run(command, check=True)
    return True


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="Generate a simplified academic-style 4x4 FDN block diagram."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "paper_assets" / "figures",
        help="Directory for rendered figure outputs.",
    )
    parser.add_argument(
        "--basename",
        default="fig_fdn_4x4_block_diagram",
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
