#!/usr/bin/env python3
"""Render the alternate schematic into .runs/. Requires librsvg (rsvg-convert)."""
from html import escape
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

W, H = 1360, 536
INK, SOFT, RULE = "#15243a", "#4d5a6b", "#d8e2ed"
BLUE, AMBER, GREEN, PURPLE = "#285e90", "#9c4a24", "#306953", "#674b93"
parts = []

def rect(x, y, w, h, fill, stroke="none", radius=0, sw=1.4):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                 f'rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

def text(x, y, value, size, color=INK, *, serif=False, bold=False, center=False):
    family = "Georgia, serif" if serif else "Helvetica Neue, Helvetica, sans-serif"
    anchor = "middle" if center else "start"
    weight = 700 if bold else 400
    parts.append(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" '
                 f'fill="{color}" font-weight="{weight}" text-anchor="{anchor}">{escape(value)}</text>')

def path(d, color, width=2, fill="none", dash=""):
    parts.append(f'<path d="{d}" stroke="{color}" stroke-width="{width}" fill="{fill}" '
                 f'stroke-linecap="round" stroke-linejoin="round" stroke-dasharray="{dash}"/>')

def dot(x, y, radius, color):
    parts.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{color}"/>')

def arrow(x1, y, x2, color):
    path(f"M{x1} {y} H{x2} m-8 -5.5 l8 5.5 l-8 5.5", color, 2.1)

def browser(x, y, scale=1, *, subscribed=False, backdrop="#f7faff", moved=False):
    parts.append(f'<g transform="translate({x} {y}) scale({scale})">')
    rect(0, 0, 374, 164, "#ffffff", "#87a7c2", 10, 1.8)
    rect(1.4, 1.4, 371.2, 23, "#e9f1f8", radius=8)
    path("M1 25 H373", "#b6c8da", 1.3)
    for i in range(3):
        dot(14+i*10, 13, 2.7, "#9fb5c9")
    rect(2, 27, 370, 135, backdrop, radius=7)
    rect(18, 47, 30, 30, "#dbeaf6", radius=7)
    path("M25 55 h16 v10 h-7 l-5 4 v-4 h-4 z", BLUE, 1.4)
    text(57, 68, "Science Forum", 19, bold=True)
    for i, length in enumerate((138, 110, 129)):
        rect(19, 91+i*14, length, 5, "#c2d1df", radius=2)
    bx, by = (168, 111) if moved else (222, 50)
    if moved:
        path("M223 50 h134 v37 h-134 z", "#9c8abb", 1.4, dash="4 4")
        path("M352 91 q0 31 -41 37", PURPLE, 1.4, dash="4 4")
    outline = AMBER if subscribed else PURPLE if moved else BLUE
    rect(bx-3, by-3, 140, 41, "none", outline, 8, 1.8)
    rect(bx, by, 134, 35, "#f7e6dc" if subscribed else "#dcebf8", radius=6)
    text(bx+67, by+23.5, "Subscribed" if subscribed else "Subscribe", 18,
         AMBER if subscribed else BLUE, bold=True, center=True)
    parts.append("</g>")

def draw():
    parts.clear()
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
    rect(0, 0, W, H, "#ffffff")
    text(50, 48, "(a) Original state", 34, serif=True)
    text(602, 48, "(b) Intervention and expected response", 34, serif=True)
    path("M50 68 H1310", RULE, 1.3)

    text(50, 142, "Goal: subscribe to this forum", 27, serif=True)
    browser(50, 168, 1.02)
    text(50, 378, "Original action", 19, SOFT)
    text(50, 410, "Click Subscribe", 31, BLUE, serif=True, bold=True)

    # Shared task and source state branch into three independently controlled variants.
    path("M444 252 H530 M530 135 V409", "#7690a8", 2.1)
    dot(530, 252, 3.8, "#7690a8")
    cases = (
        (135, AMBER, "Targeted", "Change subscription state", "Stop", "The goal is already satisfied.",
         {"subscribed": True}),
        (272, GREEN, "Unrelated", "Change only the background", "Keep the action", "The goal and control are unchanged.",
         {"backdrop": "#eaf4e9"}),
        (409, PURPLE, "Layout", "Move the control", "Move the click", "The same control is in a new place.",
         {"moved": True}),
    )
    for i, (y, color, kind, change, action, reason, kwargs) in enumerate(cases):
        if i:
            path(f"M602 {y-69} H1310", RULE, 1.05)
        arrow(530, y, 581, color)
        text(602, y-5, kind, 28, color, serif=True, bold=True)
        text(602, y+23, change, 17, SOFT)
        browser(834, y-48, 0.58, **kwargs)
        arrow(1062, y, 1090, color)
        text(1104, y-3, action, 27, color, serif=True, bold=True)
        text(1104, y+24, reason, 14, SOFT)

    path("M50 478 H1310", RULE, 1.3)
    text(680, 513, "Environment checks establish which actions remain valid in each view.",
         22, SOFT, serif=True, center=True)
    parts.append("</svg>")
    return "\n".join(parts)

def main():
    converter = shutil.which("rsvg-convert")
    if not converter:
        raise SystemExit("Install librsvg (rsvg-convert) to render the figure.")
    output = Path(__file__).resolve().parents[1] / ".runs" / "causalact-schematic.png"
    output.parent.mkdir(exist_ok=True)
    with TemporaryDirectory() as directory:
        source = Path(directory) / "overview.svg"
        source.write_text(draw(), encoding="utf-8")
        subprocess.run([converter, "-w", str(W * 2), "-h", str(H * 2),
                        "-o", str(output), str(source)], check=True)

if __name__ == "__main__":
    main()
