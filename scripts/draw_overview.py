#!/usr/bin/env python3
"""Render the project overview as a high-resolution PNG (requires rsvg-convert)."""
from pathlib import Path
from tempfile import TemporaryDirectory
from html import escape
import shutil
import subprocess

W, H = 1480, 626
INK, MUTED = "#142844", "#42546c"
BLUE = ("#245ca7", "#eef5fe", "#a9c6ec", "#dceafc")
ORANGE = ("#9b421e", "#fff4ea", "#efc3a8", "#ffe4d1")
GREEN = ("#236451", "#edf7f2", "#add7c3", "#d9ede1")
VIOLET = ("#67479c", "#f5effc", "#c9b8e6", "#eadff8")
parts = []

def add(s):
    parts.append(s)

def rect(x, y, w, h, fill, stroke="none", r=10, sw=1.3):
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

def line(x1, y1, x2, y2, color, width=2):
    add(f'<path d="M{x1} {y1} L{x2} {y2}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round"/>')

def arrow(x1, y, x2, color):
    line(x1, y, x2, y, color, 2.1)
    add(f'<path d="M{x2-7} {y-5} L{x2} {y} L{x2-7} {y+5}" fill="none" stroke="{color}" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"/>')

def circle(x, y, radius, fill):
    add(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{fill}"/>')

def label(x, y, s, size=16, color=INK, bold=False, middle=False, serif=False, tracking=0):
    anchor = ' text-anchor="middle"' if middle else ''
    face = 'Georgia, serif' if serif else 'Helvetica Neue, Helvetica, sans-serif'
    weight = "700" if bold else "400"
    add(f'<text x="{x}" y="{y}" font-family="{face}" font-size="{size}" fill="{color}" font-weight="{weight}" letter-spacing="{tracking}"{anchor}>{escape(s)}</text>')

def check(x, y, color):
    add(f'<path d="M{x-5} {y} l3.8 4 l7.3 -8" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>')

def pointer(x, y, color):
    add(f'<path d="M{x} {y} l3 16 4 -5 6 -1 Z" fill="{color}" stroke="#ffffff" stroke-width="1.3" stroke-linejoin="round"/>')

add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">')
rect(0,0,W,H,"#ffffff",r=0)
label(38,35,"CAUSALACT",15,BLUE[0],True,tracking=2.5)
label(38,80,"Respond to the evidence that changes the decision",41,INK,True,serif=True)
label(39,112,"One goal, four controlled views. Follow the subscription state; ignore the decoration; track the control.",19,MUTED)

cards = [
    (38, "01", "Original", "Baseline state", BLUE, "Subscribe", "Click Subscribe", "The goal is not yet satisfied."),
    (394, "02", "Targeted", "Change the subscription state", ORANGE, "Subscribed", "Stop: already subscribed", "The goal is already satisfied."),
    (750, "03", "Unrelated", "Change only the background", GREEN, "Subscribe", "Keep the same action", "Decorative changes leave the goal intact."),
    (1106, "04", "Layout", "Move the same control", VIOLET, "Subscribe", "Click at the new location", "The target moves; its role stays fixed."),
]
for i, (x, num, title, desc, theme, button, action, why) in enumerate(cards):
    c, fill, border, tint = theme
    rect(x,140,336,280,fill,border,r=15)
    rect(x+17,155,39,27,tint,r=6)
    label(x+36.5,174,num,13,c,True,True)
    label(x+67,176,title,25,c,True)
    label(x+18,203,desc,16,MUTED)
    wx, wy = x+18, 219
    rect(wx,wy,300,122,"#ffffff",border,r=8)
    rect(wx+1,wy+1,298,20,"#f1f5fa",r=6)
    for j,color in enumerate(("#b3c0d1","#c3cede","#d0dbe7")):
        circle(wx+12+9*j,wy+10,2.3,color)
    line(wx+1,wy+21,wx+299,wy+21,"#dfe6ef",1)
    rect(wx+2,wy+22,296,98,"#e4f2e8" if i==2 else "#fbfcff",r=4)
    circle(wx+26,wy+44,11,"#dce8f8")
    add(f'<path d="M{wx+21} {wy+43} h10 v5 h-4 l-3 2 v-2 h-3 Z" fill="#6b90bc"/>')
    label(wx+44,wy+49,"Science Forum",15,INK,True)
    for j,n in enumerate((113,89,102)):
        rect(wx+16,wy+71+j*11,n,4,"#c4d0df",r=2)
    bx = wx+167 if i != 3 else wx+137
    by = wy+38 if i != 3 else wy+75
    if i==3:
        add(f'<rect x="{wx+168}" y="{wy+34}" width="112" height="32" rx="7" fill="none" stroke="{border}" stroke-width="1.4" stroke-dasharray="4 4"/>')
        add(f'<path d="M{wx+278} {wy+62} q4 19 -20 26" fill="none" stroke="{c}" stroke-width="1.6" stroke-dasharray="4 4"/>')
    rect(bx-3,by-3,118,32,"none",c,r=8,sw=1.5)
    button_color, button_fill = (c, tint) if i==1 else (BLUE[0], BLUE[3])
    rect(bx,by,112,26,button_fill,r=5)
    label(bx+56,by+18,button,13,button_color,True,True)
    if i != 1:
        pointer(bx+92,by+17,c)
    else:
        check(bx+102,by+13,c)
    label(x+18,374,action,20,c,True)
    label(x+18,398,why,14,MUTED)

line(38,446,1442,446,"#dce5ef",1.2)
label(38,478,"FROM CONTROLLED TESTS TO SELECTIVE CORRECTIONS",14,BLUE[0],True,tracking=1.4)
bottom = [
    (38, 438, ORANGE, "I", "Environment edits", "Replay the state and validate the change."),
    (522, 436, GREEN, "A", "Action refiner", "Learn from the acceptable actions in each view."),
    (1004, 438, VIOLET, "V", "Verifier gate", "Pass through the base action or apply a correction."),
]
for x, w, (c, fill, border, tint), letter, title, desc in bottom:
    rect(x,493,w,84,fill,border,r=12)
    rect(x+16,512,45,45,tint,r=10)
    label(x+38.5,542,letter,23,c,True,True)
    label(x+75,526,title,21,c,True)
    label(x+75,551,desc,15,MUTED)
arrow(483,535,514,BLUE[0])
arrow(965,535,996,BLUE[0])
label(740,607,"Illustrative task: subscribe to a forum. The environment determines which actions are acceptable.",14,MUTED,False,True)
add('</svg>')

def main():
    converter = shutil.which("rsvg-convert")
    if not converter:
        raise SystemExit("Install librsvg (rsvg-convert) to render this figure.")
    output = Path(__file__).resolve().parents[1] / "assets" / "causalact-overview.png"
    output.parent.mkdir(exist_ok=True)
    with TemporaryDirectory() as directory:
        source = Path(directory) / "overview.svg"
        source.write_text("\n".join(parts), encoding="utf-8")
        subprocess.run([converter, "-w", str(W * 2), "-h", str(H * 2),
                        "-o", str(output), str(source)], check=True)

if __name__ == "__main__":
    main()
