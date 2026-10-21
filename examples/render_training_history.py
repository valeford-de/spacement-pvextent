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
    parent = json.loads((research / "baseline-negative-comparison-v1.json").read_text())
    continuation = json.loads((research / "training-extension-v1.json").read_text())
    first, second = parent["candidate"], continuation["run"]
    assert first["dataset_fingerprint"] == second["dataset_fingerprint"]
    assert first["last_checkpoint_sha256"] == continuation["parent"]["checkpoint_sha256"]
    assert first["train_samples"] == second["train_samples"] == 3293
    assert first["val_samples"] == second["val_samples"] == 278
    rows = first["epochs"] + second["history"]
    epochs = [row["epoch"] for row in rows]
    assert epochs == [1, 2, 3, 4, 5, 6]
    best_epoch = second["best_epoch"]
    assert best_epoch == max(rows, key=lambda row: row["validation"]["iou"])["epoch"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "text.color": "#25312d",
        "axes.labelcolor": "#4a5752",
        "xtick.color": "#53615a",
        "ytick.color": "#53615a",
        "svg.hashsalt": "pvextent-training-history-v1",
    })
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), dpi=100)
    fig.subplots_adjust(left=0.075, right=0.975, top=0.71, bottom=0.235, wspace=0.25)
    fig.patch.set_facecolor("white")
    fig.text(0.075, 0.91, "Six-epoch baseline training", fontsize=17, weight="bold")
    fig.text(0.075, 0.835,
             "3,293 training crops  /  278 validation crops  /  unchanged dataset",
             fontsize=10.5, color="#53615a")

    for ax in axes:
        ax.set_facecolor("white")
        ax.set_xlim(0.8, 6.2)
        ax.set_xticks(epochs)
        ax.set_xlabel("Epoch", labelpad=9)
        ax.tick_params(axis="both", length=0, pad=7, labelsize=10)
        ax.grid(axis="y", color="#e4e9e6", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#ccd5cf")

    axes[0].set_title("Training loss", loc="left", fontsize=12, weight="bold", pad=18)
    axes[0].plot(epochs, [row["train_loss"] for row in rows],
                 color="#394e5c", linewidth=2, marker="o", markersize=4.5)
    axes[0].set_ylim(0.10, 0.40)
    axes[0].set_yticks([0.1, 0.2, 0.3, 0.4])

    iou = [row["validation"]["iou"] for row in rows]
    axes[1].set_title("Internal validation IoU", loc="left", fontsize=12,
                      weight="bold", pad=18)
    axes[1].plot(epochs, iou, color="#009e73", linewidth=2, marker="o", markersize=4.5)
    axes[1].set_ylim(0.885, 0.933)
    axes[1].set_yticks([0.89, 0.90, 0.91, 0.92, 0.93])
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    best_iou = rows[best_epoch - 1]["validation"]["iou"]
    axes[1].plot([best_epoch], [best_iou], marker="o", color="#009e73", markersize=7)
    axes[1].annotate(f"Best: {best_iou:.2%}", xy=(best_epoch, best_iou),
                     xytext=(-10, 13), textcoords="offset points", ha="right",
                     color="#007b59", fontsize=10, weight="bold")

    fig.text(0.075, 0.07,
             "Recorded epoch values; straight lines join measurements. New NRW labels were not used.",
             fontsize=9.5, color="#53615a")
    output = root / "docs" / "assets" / "training-history.svg"
    fig.savefig(output, format="svg", metadata={
        "Date": None,
        "Title": "Spacement PVExtent: recorded six-epoch training history",
        "Description": "Epochs 1–3 from baseline-negative-comparison-v1.json; epochs 4–6 "
                       "from training-extension-v1.json. Internal validation only. "
                       "No image, mask or checkpoint was loaded to render this chart.",
        "Creator": "Spacement PVExtent / Valeford",
    })
    plt.close(fig)
    print(output)
    for row in rows:
        print(f"epoch={row['epoch']} train_loss={row['train_loss']:.12f} "
              f"val_iou={row['validation']['iou']:.12f}")


if __name__ == "__main__":
    main()
