"""Builds the two schematic figures for Section 3 (grayscale, >= 9 pt text, PNG at 400 dpi).

Run from anywhere:  python paper/figures/make_diagrams.py
Outputs (next to this script): pipeline.png, movement_step.png
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
})
OUT = os.path.dirname(os.path.abspath(__file__))
FONT = 9


# ---------------------------------------------------------------- Figure A: pipeline (full width)
def pipeline():
    fig, ax = plt.subplots(figsize=(7.0, 1.95))
    ax.set_xlim(0, 7.0)
    ax.set_ylim(0, 1.95)
    ax.axis("off")

    boxes = [
        ("Data", ["Telemetry fixes", "Vehicular deaths", "Road network"]),
        ("Graph", ["Nodes: segments", "Edges: adjacency", "and movement"]),
        ("Inputs", ["13 features $x_i$", "Label $y_i$", "(death within $b$)"]),
        ("Models", ["GraphSAGE", "Adjacency-only", "Random forest", "XGBoost, KDE"]),
        ("Evaluation", ["Spatial 5-fold CV", "AUC-PR", "Top-$p$% recall", "Brier score"]),
    ]
    n = len(boxes)
    w, gap = 1.22, 0.225
    x0 = (7.0 - (n * w + (n - 1) * gap)) / 2
    y, h = 0.12, 1.71
    shades = ["0.93", "0.85", "0.93", "0.85", "0.93"]

    for i, (title, lines) in enumerate(boxes):
        x = x0 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.06",
                                    facecolor=shades[i], edgecolor="black", linewidth=0.9))
        ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="center", fontsize=FONT, fontweight="bold")
        ax.plot([x + 0.08, x + w - 0.08], [y + h - 0.36, y + h - 0.36], color="black", linewidth=0.6)
        for j, line in enumerate(lines):
            ax.text(x + w / 2, y + h - 0.6 - j * 0.27, line, ha="center", va="center", fontsize=FONT)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.02, y + h / 2), (x + w + gap - 0.02, y + h / 2),
                                         arrowstyle="-|>", mutation_scale=9, color="black", linewidth=0.9))
    fig.savefig(os.path.join(OUT, "pipeline.png"), dpi=400, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


# ---------------------------------------------------------------- Figure B: movement step (one column)
def movement_step():
    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    xi, xj = 3.6, 7.0                                   # the two road segments (vertical)
    ax.plot([xi, xi], [0.4, 5.3], color="black", linewidth=3.2, solid_capstyle="butt")
    ax.plot([xj, xj], [0.4, 5.3], color="black", linewidth=3.2, solid_capstyle="butt")
    ax.text(xi, 0.0, "segment $v_i$", ha="center", va="top", fontsize=FONT)
    ax.text(xj, 0.0, "segment $v_j$", ha="center", va="top", fontsize=FONT)

    p0, p1 = (0.8, 1.9), (9.3, 3.4)                     # two consecutive fixes
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], linestyle="--", color="black", linewidth=1.1, zorder=2)
    for p, lab, dy in ((p0, "fix $t$", -0.75), (p1, "fix $t{+}1$", -0.75)):
        ax.scatter(*p, s=36, facecolor="white", edgecolor="black", linewidth=1.0, zorder=4)
        ax.text(p[0], p[1] + dy, lab, ha="center", va="top", fontsize=FONT)

    def y_at(x):                                         # point on the step at a given x
        return p0[1] + (p1[1] - p0[1]) * (x - p0[0]) / (p1[0] - p0[0])
    for x in (xi, xj):
        ax.scatter(x, y_at(x), marker="X", s=50, facecolor="white", edgecolor="black", linewidth=0.9, zorder=5)

    ax.text(5.3, y_at(5.3) + 0.55, "movement step", ha="center", va="bottom", fontsize=FONT, rotation=9)

    # movement edge drawn above the two segments
    ax.add_patch(FancyArrowPatch((xi, 5.55), (xj, 5.55), connectionstyle="arc3,rad=-0.55",
                                 arrowstyle="<->", mutation_scale=9, color="black", linewidth=1.0))
    ax.text((xi + xj) / 2, 7.0, "movement edge $w_{ij}\\!+\\!1$", ha="center", va="bottom", fontsize=FONT)

    fig.savefig(os.path.join(OUT, "movement_step.png"), dpi=400, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    pipeline()
    movement_step()
    print("wrote pipeline.png and movement_step.png to", OUT)
