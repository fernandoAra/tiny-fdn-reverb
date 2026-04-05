#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import math
import shutil
import subprocess
from pathlib import Path


WIDTH = 1500
HEIGHT = 820

BG = "#ffffff"
STROKE = "#222222"
MUTED = "#5a5a5a"
GRID = "#d7d7d7"
SHORT_COLOR = "#2c7fb8"
LONG_COLOR = "#d95f02"
LEGEND_BG = "#fbfbfb"

PANEL_Y = 55
PANEL_H = 700
PANEL_W = 660
PANEL_A_X = 70
PANEL_B_X = 770

MARGIN_LEFT = 84
MARGIN_RIGHT = 28
MARGIN_TOP = 85
MARGIN_BOTTOM = 92

RT60 = 1.0
SHORT_DELAY = 0.07
LONG_DELAY = 0.18
XMAX = 1.05


def loop_gain(delay_s: float, rt60_s: float) -> float:
    return 10.0 ** (-3.0 * delay_s / rt60_s)


SHORT_GAIN = loop_gain(SHORT_DELAY, RT60)
LONG_GAIN = loop_gain(LONG_DELAY, RT60)


def echo_train(delay_s: float, gain: float, tmax: float) -> list[tuple[float, float]]:
    samples: list[tuple[float, float]] = []
    n = 0
    while True:
        t = n * delay_s
        if t > tmax + 1e-9:
            break
        samples.append((t, gain**n))
        n += 1
    return samples


SHORT_POINTS = echo_train(SHORT_DELAY, SHORT_GAIN, XMAX)
LONG_POINTS = echo_train(LONG_DELAY, LONG_GAIN, XMAX)


def line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    color: str = STROKE,
    stroke_width: float = 2.5,
    dash: str | None = None,
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'fill="none" stroke="{color}" stroke-width="{stroke_width}" '
        f'stroke-linecap="round" stroke-linejoin="round"{dash_attr} />'
    )


def rect(
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    fill: str = BG,
    stroke: str = STROKE,
    stroke_width: float = 2.2,
    rx: float = 0,
) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" />'
    )


def circle(x: float, y: float, r: float, *, fill: str = BG, stroke: str = STROKE, stroke_width: float = 2.0) -> str:
    return (
        f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{stroke_width}" />'
    )


def text(
    x: float,
    y: float,
    content: str,
    *,
    size: int = 26,
    anchor: str = "middle",
    weight: str = "400",
    color: str = STROKE,
    italic: bool = False,
    rotate: float | None = None,
) -> str:
    style = "italic" if italic else "normal"
    transform = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-size="{size}" font-weight="{weight}" fill="{color}" '
        f'font-style="{style}"{transform}>{html.escape(content)}</text>'
    )


def multiline_text(
    x: float,
    y: float,
    lines: list[str],
    *,
    size: int = 22,
    line_gap: int = 28,
    anchor: str = "middle",
    weight: str = "400",
    color: str = STROKE,
) -> str:
    parts = [
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}">'
    ]
    for idx, content in enumerate(lines):
        dy = "0" if idx == 0 else str(line_gap)
        parts.append(f'<tspan x="{x}" dy="{dy}">{html.escape(content)}</tspan>')
    parts.append("</text>")
    return "".join(parts)


def rich_text(
    x: float,
    y: float,
    spans: list[dict[str, object]],
    *,
    size: int = 22,
    anchor: str = "middle",
    weight: str = "400",
    color: str = STROKE,
) -> str:
    parts = [
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}" xml:space="preserve">'
    ]
    for span in spans:
        attrs = []
        if span.get("italic"):
            attrs.append('font-style="italic"')
        if "size" in span:
            attrs.append(f'font-size="{span["size"]}"')
        if "baseline_shift" in span:
            attrs.append(f'baseline-shift="{span["baseline_shift"]}"')
        parts.append(f'<tspan {" ".join(attrs)}>{html.escape(str(span["text"]))}</tspan>')
    parts.append("</text>")
    return "".join(parts)


def square_marker(x: float, y: float, size: float, *, fill: str = BG, stroke: str = STROKE, stroke_width: float = 2.0) -> str:
    half = size / 2.0
    return rect(x - half, y - half, size, size, fill=fill, stroke=stroke, stroke_width=stroke_width)


def axis_box(panel_x: float) -> tuple[float, float, float, float]:
    x0 = panel_x + MARGIN_LEFT
    y0 = PANEL_Y + MARGIN_TOP
    w = PANEL_W - MARGIN_LEFT - MARGIN_RIGHT
    h = PANEL_H - MARGIN_TOP - MARGIN_BOTTOM
    return x0, y0, w, h


def map_x(value: float, x0: float, w: float) -> float:
    return x0 + (value / XMAX) * w


def map_y_amp(value: float, y0: float, h: float) -> float:
    return y0 + h - value * h


def map_y_db(value: float, y0: float, h: float, db_min: float = -70.0, db_max: float = 0.0) -> float:
    frac = (value - db_min) / (db_max - db_min)
    return y0 + h - frac * h


def panel_a() -> str:
    panel_x = PANEL_A_X
    x0, y0, w, h = axis_box(panel_x)
    parts: list[str] = []
    parts.append(rect(panel_x, PANEL_Y, PANEL_W, PANEL_H, fill=BG, stroke=STROKE, stroke_width=2.2))
    parts.append(text(panel_x + 38, PANEL_Y + 34, "(a)", size=24, weight="600"))
    parts.append(
        multiline_text(
            panel_x + PANEL_W / 2 + 18,
            PANEL_Y + 28,
            ["Different delay lengths require", "different per-loop gains"],
            size=22,
            line_gap=24,
            weight="500",
        )
    )
    legend_y = PANEL_Y + 76
    parts.append(line(panel_x + 116, legend_y, panel_x + 144, legend_y, color=SHORT_COLOR, stroke_width=3.2))
    parts.append(circle(panel_x + 130, legend_y, 5.0, fill=BG, stroke=SHORT_COLOR, stroke_width=2.4))
    parts.append(text(panel_x + 154, legend_y, "Short: 70 ms, g ≈ 0.62", size=17, anchor="start", color=MUTED))
    parts.append(line(panel_x + 360, legend_y, panel_x + 388, legend_y, color=LONG_COLOR, stroke_width=3.2))
    parts.append(square_marker(panel_x + 374, legend_y, 10, fill=BG, stroke=LONG_COLOR, stroke_width=2.1))
    parts.append(text(panel_x + 398, legend_y, "Long: 180 ms, g ≈ 0.29", size=17, anchor="start", color=MUTED))
    parts.append(text(panel_x + PANEL_W / 2, PANEL_Y + 102, "Both cases target RT60 = 1.0 s", size=17, weight="500", color=MUTED))

    for t in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        x = map_x(t, x0, w)
        parts.append(line(x, y0, x, y0 + h, color=GRID, stroke_width=1.2))
        parts.append(text(x, y0 + h + 28, f"{t:.1f}", size=18, color=MUTED))

    for a in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        y = map_y_amp(a, y0, h)
        parts.append(line(x0, y, x0 + w, y, color=GRID, stroke_width=1.2))
        parts.append(text(x0 - 14, y, f"{a:.1f}", size=18, anchor="end", color=MUTED))

    parts.append(line(x0, y0, x0, y0 + h, stroke_width=2.2))
    parts.append(line(x0, y0 + h, x0 + w, y0 + h, stroke_width=2.2))
    parts.append(text(x0 + w / 2, y0 + h + 62, "Time (s)", size=22, weight="500"))
    parts.append(text(x0 - 62, y0 + h / 2, "Amplitude", size=22, weight="500", rotate=-90))

    for t, amp in SHORT_POINTS:
        x = map_x(t, x0, w)
        y = map_y_amp(amp, y0, h)
        parts.append(line(x, map_y_amp(0.0, y0, h), x, y, color=SHORT_COLOR, stroke_width=2.2))
        parts.append(circle(x, y, 5.5, fill=BG, stroke=SHORT_COLOR, stroke_width=2.6))

    for t, amp in LONG_POINTS:
        x = map_x(t, x0, w)
        y = map_y_amp(amp, y0, h)
        parts.append(line(x, map_y_amp(0.0, y0, h), x, y, color=LONG_COLOR, stroke_width=2.8))
        parts.append(square_marker(x, y, 11, fill=BG, stroke=LONG_COLOR, stroke_width=2.2))

    short_annot_x = map_x(SHORT_POINTS[2][0], x0, w) + 16
    short_annot_y = map_y_amp(SHORT_POINTS[2][1], y0, h) - 22
    parts.append(text(short_annot_x, short_annot_y, "Short delay", size=19, anchor="start", color=SHORT_COLOR, weight="500"))

    long_annot_x = map_x(LONG_POINTS[1][0], x0, w) + 18
    long_annot_y = map_y_amp(LONG_POINTS[1][1], y0, h) - 20
    parts.append(text(long_annot_x, long_annot_y, "Long delay", size=19, anchor="start", color=LONG_COLOR, weight="500"))

    return "\n".join(parts)


def panel_b() -> str:
    panel_x = PANEL_B_X
    x0, y0, w, h = axis_box(panel_x)
    db_min = -70.0
    parts: list[str] = []
    parts.append(rect(panel_x, PANEL_Y, PANEL_W, PANEL_H, fill=BG, stroke=STROKE, stroke_width=2.2))
    parts.append(text(panel_x + 38, PANEL_Y + 34, "(b)", size=24, weight="600"))
    parts.append(text(panel_x + PANEL_W / 2, PANEL_Y + 34, "Same RT60 in elapsed time", size=24, weight="500"))
    legend_y = PANEL_Y + 76
    parts.append(line(panel_x + 102, legend_y, panel_x + 130, legend_y, color=SHORT_COLOR, stroke_width=3.2))
    parts.append(circle(panel_x + 116, legend_y, 5.0, fill=BG, stroke=SHORT_COLOR, stroke_width=2.4))
    parts.append(text(panel_x + 140, legend_y, "Short-delay samples", size=17, anchor="start", color=MUTED))
    parts.append(line(panel_x + 310, legend_y, panel_x + 338, legend_y, color=LONG_COLOR, stroke_width=3.2))
    parts.append(square_marker(panel_x + 324, legend_y, 10, fill=BG, stroke=LONG_COLOR, stroke_width=2.1))
    parts.append(text(panel_x + 348, legend_y, "Long-delay samples", size=17, anchor="start", color=MUTED))
    parts.append(line(panel_x + 494, legend_y, panel_x + 522, legend_y, color=STROKE, stroke_width=2.0, dash="7 6"))
    parts.append(text(panel_x + 532, legend_y, "Common RT60 target", size=17, anchor="start", color=MUTED))

    for t in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        x = map_x(t, x0, w)
        parts.append(line(x, y0, x, y0 + h, color=GRID, stroke_width=1.2))
        parts.append(text(x, y0 + h + 28, f"{t:.1f}", size=18, color=MUTED))

    for db in [0.0, -20.0, -40.0, -60.0]:
        y = map_y_db(db, y0, h, db_min=db_min)
        parts.append(line(x0, y, x0 + w, y, color=GRID, stroke_width=1.2))
        label = "0" if db == 0 else f"{int(db)}"
        parts.append(text(x0 - 16, y, label, size=18, anchor="end", color=MUTED))

    parts.append(line(x0, y0, x0, y0 + h, stroke_width=2.2))
    parts.append(line(x0, y0 + h, x0 + w, y0 + h, stroke_width=2.2))
    parts.append(text(x0 + w / 2, y0 + h + 62, "Time (s)", size=22, weight="500"))
    parts.append(text(x0 - 62, y0 + h / 2, "Level (dB)", size=22, weight="500", rotate=-90))

    rt60_x = map_x(RT60, x0, w)
    minus60_y = map_y_db(-60.0, y0, h, db_min=db_min)
    parts.append(line(rt60_x, y0, rt60_x, y0 + h, color=MUTED, stroke_width=1.8, dash="7 6"))
    parts.append(line(x0, minus60_y, x0 + w, minus60_y, color=MUTED, stroke_width=1.8, dash="7 6"))
    parts.append(text(rt60_x, y0 - 14, "RT60", size=19, weight="500", color=MUTED))
    parts.append(text(x0 + w - 4, minus60_y - 14, "-60 dB", size=18, anchor="end", color=MUTED))

    target_points = []
    step = 0.01
    t = 0.0
    while t <= XMAX + 1e-9:
        target_points.append((map_x(t, x0, w), map_y_db(-60.0 * t / RT60, y0, h, db_min=db_min)))
        t += step
    path = " ".join(f"L {x:.2f} {y:.2f}" if idx else f"M {x:.2f} {y:.2f}" for idx, (x, y) in enumerate(target_points))
    parts.append(
        f'<path d="{path}" fill="none" stroke="{STROKE}" stroke-width="2.0" stroke-dasharray="8 6" />'
    )
    for t, amp in SHORT_POINTS:
        x = map_x(t, x0, w)
        db = 20.0 * math.log10(max(amp, 1e-9))
        y = map_y_db(db, y0, h, db_min=db_min)
        parts.append(circle(x, y, 5.2, fill=BG, stroke=SHORT_COLOR, stroke_width=2.6))

    for t, amp in LONG_POINTS:
        x = map_x(t, x0, w)
        db = 20.0 * math.log10(max(amp, 1e-9))
        y = map_y_db(db, y0, h, db_min=db_min)
        parts.append(square_marker(x, y, 10.5, fill=BG, stroke=LONG_COLOR, stroke_width=2.2))

    return "\n".join(parts)


def svg_markup() -> str:
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
                f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">'
            ),
            "<title id=\"title\">Homogeneous decay and RT60 control in a tiny FDN</title>",
            (
                "<desc id=\"desc\">Two-panel conceptual figure explaining that shorter and longer "
                "delay lines require different per-loop gains to achieve the same RT60 target in seconds.</desc>"
            ),
            "<defs>",
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
    )


def convert_with_rsvg(svg_path: Path, output_path: Path, fmt: str) -> bool:
    rsvg = shutil.which("rsvg-convert")
    if not rsvg:
        return False
    subprocess.run([rsvg, "-f", fmt, "-o", str(output_path), str(svg_path)], check=True)
    return True


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="Generate the homogeneous-decay RT60 conceptual figure for Section 2.4."
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "paper_assets" / "figures",
        help="Directory for rendered figure outputs.",
    )
    parser.add_argument(
        "--basename",
        default="fig_homogeneous_decay_rt60_explanation",
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
