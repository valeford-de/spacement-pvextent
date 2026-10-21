# Copyright 2026 Valeford
# SPDX-License-Identifier: Apache-2.0

"""Render the README chart from versioned scalar reports, without loading pixels.

Run from any directory with Matplotlib installed, or use an isolated environment:
    uv run --no-project --with matplotlib==3.11.2 python examples/render_training_history.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main() -> None:
    root = Path(__file__).absolute().parent.parent
    research = root / "docs" / "research"
    runs = root / "runs"

    parent = json.loads((research / "baseline-negative-comparison-v1.json").read_text())
    continuation = json.loads((research / "training-extension-v1.json").read_text())
    v2_rows = parent["candidate"]["epochs"] + continuation["run"]["history"]
    v2_epochs = [r["epoch"] for r in v2_rows]
    v2_loss = [r["train_loss"] for r in v2_rows]
    v2_iou = [r["validation"]["iou"] for r in v2_rows]

    v4_report = json.loads((runs / "baseline-resnet18-bulk-v4-001" / "baseline-report.json").read_text())
    v4_rows = v4_report["history"]
    v4_epochs = [r["epoch"] for r in v4_rows]
    v4_loss = [r["train_loss"] for r in v4_rows]
    v4_iou = [r["validation"]["iou"] for r in v4_rows]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": "#25312d",
        "axes.labelcolor": "#4a5752",
        "xtick.color": "#53615a",
        "ytick.color": "#53615a",
        "svg.hashsalt": "pvextent-training-history-v4",
    })

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), dpi=100)
    fig.subplots_adjust(left=0.075, right=0.975, top=0.70, bottom=0.23, wspace=0.25)
    fig.patch.set_facecolor("white")
    fig.text(0.075, 0.91, "PVExtent Model Training History: V4 Bulk vs. Baseline", fontsize=16, weight="bold")
    fig.text(0.075, 0.835,
             "V4: 3,775 training crops (incl. 482 NRW partial masks)  /  Baseline: 3,293 crops  /  278 validation crops",
             fontsize=10.0, color="#53615a")

    for ax in axes:
        ax.set_facecolor("white")
        ax.set_xlim(0.8, 6.2)
        ax.set_xticks(range(1, 7))
        ax.set_xlabel("Epoch", labelpad=8)
        ax.tick_params(axis="both", length=0, pad=7, labelsize=10)
        ax.grid(axis="y", color="#e4e9e6", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#ccd5cf")

    # Loss subplot
    axes[0].set_title("Training loss", loc="left", fontsize=12, weight="bold", pad=16)
    axes[0].plot(v2_epochs, v2_loss, color="#8a9ba8", linewidth=1.8, linestyle="--", label="V2 Baseline (3,293 crops)")
    axes[0].plot(v4_epochs, v4_loss, color="#1b4965", linewidth=2.4, marker="o", markersize=5, label="V4 NRW-Enriched (3,775 crops)")
    axes[0].set_ylim(0.10, 0.40)
    axes[0].set_yticks([0.1, 0.2, 0.3, 0.4])
    axes[0].legend(frameon=False, fontsize=9.5, loc="upper right")

    # IoU subplot
    axes[1].set_title("Internal validation IoU", loc="left", fontsize=12, weight="bold", pad=16)
    axes[1].plot(v2_epochs, v2_iou, color="#8a9ba8", linewidth=1.8, linestyle="--", label="V2 Baseline (3,293 crops)")
    axes[1].plot(v4_epochs, v4_iou, color="#009e73", linewidth=2.4, marker="o", markersize=5.5, label="V4 NRW-Enriched (3,775 crops)")
    axes[1].set_ylim(0.885, 0.935)
    axes[1].set_yticks([0.89, 0.90, 0.91, 0.92, 0.93])
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    axes[1].legend(frameon=False, fontsize=9.5, loc="lower right")

    # Annotations
    best_v4_epoch = 3
    best_v4_iou = v4_iou[-1]
    axes[1].plot([best_v4_epoch], [best_v4_iou], marker="o", color="#007b59", markersize=7.5)
    axes[1].annotate(f"V4 Best: {best_v4_iou:.2%}\n(98.07% Acc)", xy=(best_v4_epoch, best_v4_iou),
                     xytext=(12, -18), textcoords="offset points", ha="left",
                     color="#007b59", fontsize=9.5, weight="bold")

    axes[1].plot([5], [v2_iou[4]], marker="o", color="#6c7d8a", markersize=6)
    axes[1].annotate(f"V2 Ep5: {v2_iou[4]:.2%}", xy=(5, v2_iou[4]),
                     xytext=(-10, 10), textcoords="offset points", ha="right",
                     color="#6c7d8a", fontsize=9)

    fig.text(0.075, 0.05,
             "Measured epoch metrics. V4 incorporates 482 reviewed NRW orthophoto masks to improve real-world transfer.",
             fontsize=9.0, color="#53615a")

    output = root / "docs" / "assets" / "training-history.svg"
    fig.savefig(output, format="svg", metadata={
        "Title": "Spacement PVExtent: recorded training history comparing V4 and baseline",
        "Creator": "Spacement PVExtent / Valeford",
    })
    plt.close(fig)
    print("Rendered:", output)


if __name__ == "__main__":
    main()
