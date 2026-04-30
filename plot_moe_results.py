#!/usr/bin/env python3
"""
Generate figures for synthetic MoE sweep results.

Usage:
    .venv/bin/python plot_moe_results.py logs/slurm-moe-12790884
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TOPOLOGY_ORDER = [
    "Circulant {1,5,17}",
    "Torus 4x4x4",
    "Mesh 4x4x4",
    "Ring 64",
]
PLACEMENT_ORDER = ["clustered", "spread", "topology_aware"]
TOPOLOGY_LABELS = {
    "Circulant {1,5,17}": "Circulant\n{1,5,17}",
    "Torus 4x4x4": "Torus\n4x4x4",
    "Mesh 4x4x4": "Mesh\n4x4x4",
    "Ring 64": "Ring\n64",
}
PLACEMENT_LABELS = {
    "clustered": "Clustered",
    "spread": "Spread",
    "topology_aware": "Topology-aware",
}
PLACEMENT_COLORS = {
    "clustered": "#D45A5A",
    "spread": "#4E79A7",
    "topology_aware": "#59A14F",
}
PHASE_COLORS = {
    "dispatch": "#4E79A7",
    "combine": "#F28E2B",
}
BG_COLOR = "#FAFAFA"
GRID_COLOR = "#E5E5E5"
TEXT_COLOR = "#222222"


def _safe_path_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._-") or "unnamed"


def _load_results(path: Path) -> tuple[dict, Path]:
    if path.is_dir():
        result_path = path / "moe_results.json"
    else:
        result_path = path
    if not result_path.exists():
        raise SystemExit(f"MoE results JSON not found: {result_path}")
    with result_path.open(encoding="utf-8") as f:
        return json.load(f), result_path


def _default_out_dir(result_path: Path) -> Path:
    parent = result_path.parent
    if parent.name.startswith("slurm-moe-"):
        return Path("figures") / f"current_moe_{parent.name}"
    return Path("figures") / f"moe_{_safe_path_part(parent.name)}"


def _style_ax(ax, title=None, ylabel=None, xlabel=None):
    ax.set_facecolor(BG_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BBBBBB")
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT_COLOR, pad=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=TEXT_COLOR)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=TEXT_COLOR)


def _ordered_workloads(results):
    seen = {}
    for result in results:
        workload = result["workload"]["name"]
        seen[workload] = result["workload"]
    return list(seen)


def _ordered_topologies(results):
    present = {result["topology"] for result in results}
    ordered = [topology for topology in TOPOLOGY_ORDER if topology in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _result_index(results):
    return {
        (
            result["workload"]["name"],
            result["topology"],
            result["placement_strategy"],
        ): result
        for result in results
    }


def _best_result(entries):
    return min(entries, key=lambda result: result["total_latency"])


def _best_by_workload_topology(results):
    grouped = defaultdict(list)
    for result in results:
        grouped[(result["workload"]["name"], result["topology"])].append(result)
    return {key: _best_result(entries) for key, entries in grouped.items()}


def fig_latency_by_topology(data, out_dir: Path) -> Path:
    results = data["results"]
    workloads = _ordered_workloads(results)
    topologies = _ordered_topologies(results)
    index = _result_index(results)

    ncols = 2
    nrows = (len(workloads) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.6 * nrows), squeeze=False)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "MoE Token Dispatch Latency by Topology and Expert Placement",
        fontsize=16,
        fontweight="bold",
        color=TEXT_COLOR,
        y=0.98,
    )

    x = np.arange(len(topologies))
    width = 0.24
    offsets = np.linspace(-width, width, len(PLACEMENT_ORDER))
    for ax, workload in zip(axes.flat, workloads):
        for placement, offset in zip(PLACEMENT_ORDER, offsets):
            values = [
                index[(workload, topology, placement)]["total_latency"] * 1e3
                for topology in topologies
            ]
            ax.bar(
                x + offset,
                values,
                width=width,
                label=PLACEMENT_LABELS[placement],
                color=PLACEMENT_COLORS[placement],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([TOPOLOGY_LABELS.get(t, t) for t in topologies])
        ax.set_yscale("log")
        _style_ax(ax, title=workload, ylabel="Latency (ms, log scale)")

    for ax in axes.flat[len(workloads):]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    path = out_dir / "1_moe_latency_by_topology.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fig_placement_speedup(data, out_dir: Path) -> Path:
    results = data["results"]
    workloads = _ordered_workloads(results)
    topologies = _ordered_topologies(results)
    index = _result_index(results)

    speedups = np.zeros((len(workloads), len(topologies)))
    annotations = []
    for row, workload in enumerate(workloads):
        annotation_row = []
        for col, topology in enumerate(topologies):
            clustered = index[(workload, topology, "clustered")]
            best = min(
                (index[(workload, topology, placement)] for placement in PLACEMENT_ORDER),
                key=lambda result: result["total_latency"],
            )
            speedup = clustered["total_latency"] / best["total_latency"]
            speedups[row, col] = speedup
            annotation_row.append(
                f"{speedup:.2f}x\n{PLACEMENT_LABELS[best['placement_strategy']]}"
            )
        annotations.append(annotation_row)

    fig, ax = plt.subplots(figsize=(12, 5.8))
    fig.patch.set_facecolor("white")
    im = ax.imshow(speedups, cmap="YlGnBu", vmin=1.0, aspect="auto")
    ax.set_title(
        "Best Expert Placement Speedup vs Clustered",
        fontsize=15,
        fontweight="bold",
        color=TEXT_COLOR,
        pad=14,
    )
    ax.set_xticks(np.arange(len(topologies)))
    ax.set_xticklabels([TOPOLOGY_LABELS.get(t, t).replace("\n", " ") for t in topologies])
    ax.set_yticks(np.arange(len(workloads)))
    ax.set_yticklabels(workloads)
    ax.tick_params(labelsize=9)
    for row in range(len(workloads)):
        for col in range(len(topologies)):
            color = "white" if speedups[row, col] > 3.0 else TEXT_COLOR
            ax.text(
                col,
                row,
                annotations[row][col],
                ha="center",
                va="center",
                fontsize=8,
                color=color,
                fontweight="bold",
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.88)
    cbar.set_label("Speedup over clustered placement", fontsize=10)
    fig.tight_layout()
    path = out_dir / "2_moe_placement_speedup.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fig_phase_breakdown(data, out_dir: Path) -> Path:
    results = data["results"]
    workloads = _ordered_workloads(results)
    topologies = _ordered_topologies(results)
    best = _best_by_workload_topology(results)

    ncols = 2
    nrows = (len(workloads) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4.5 * nrows), squeeze=False)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "MoE Dispatch and Combine Latency at Best Placement",
        fontsize=16,
        fontweight="bold",
        color=TEXT_COLOR,
        y=0.98,
    )

    x = np.arange(len(topologies))
    for ax, workload in zip(axes.flat, workloads):
        dispatch_values = []
        combine_values = []
        labels = []
        for topology in topologies:
            result = best[(workload, topology)]
            phase_by_name = {phase["phase"]: phase for phase in result["per_phase"]}
            dispatch_values.append(phase_by_name.get("dispatch", {}).get("latency", 0.0) * 1e3)
            combine_values.append(phase_by_name.get("combine", {}).get("latency", 0.0) * 1e3)
            labels.append(
                f"{TOPOLOGY_LABELS.get(topology, topology)}\n"
                f"{PLACEMENT_LABELS[result['placement_strategy']]}"
            )
        ax.bar(
            x,
            dispatch_values,
            color=PHASE_COLORS["dispatch"],
            label="Dispatch",
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.bar(
            x,
            combine_values,
            bottom=dispatch_values,
            color=PHASE_COLORS["combine"],
            label="Combine",
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        _style_ax(ax, title=workload, ylabel="Latency (ms)")

    for ax in axes.flat[len(workloads):]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    path = out_dir / "3_moe_phase_breakdown.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _format_bytes(value: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(value)
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def fig_workload_setup(data, out_dir: Path) -> Path:
    results = data["results"]
    workloads = []
    seen = set()
    for result in results:
        workload = result["workload"]
        name = workload["name"]
        if name in seen:
            continue
        seen.add(name)
        traffic = result["traffic_summary"]
        workloads.append(
            [
                name,
                str(workload["num_experts"]),
                str(workload["top_k"]),
                f"{workload['local_fraction']:.0%}",
                f"{workload['tokens_per_chip']:,}",
                f"{workload['hidden_size']:,}",
                _format_bytes(workload["hidden_size"] * workload["bytes_per_value"]),
                _format_bytes(traffic["logical_dispatch_bytes"]),
            ]
        )

    columns = [
        "Workload",
        "Experts",
        "Top-k",
        "Local\nbias",
        "Tokens\n/chip",
        "Hidden",
        "Token\nbytes",
        "Logical dispatch\nbytes",
    ]

    fig, ax = plt.subplots(figsize=(16, 4.8))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    ax.set_title(
        "Synthetic MoE Workload Setup",
        fontsize=16,
        fontweight="bold",
        color=TEXT_COLOR,
        pad=18,
    )
    table = ax.table(
        cellText=workloads,
        colLabels=columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.22, 0.08, 0.07, 0.08, 0.10, 0.10, 0.10, 0.16],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.7)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if row == 0:
            cell.set_facecolor("#E9EEF5")
            cell.set_text_props(weight="bold", color=TEXT_COLOR)
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 else "#F7F7F7")
    path = out_dir / "4_moe_workload_setup.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Plot synthetic MoE sweep results.")
    parser.add_argument("results", type=Path, help="MoE run directory or moe_results.json")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output figure directory. Defaults under figures/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data, result_path = _load_results(args.results)
    out_dir = args.out_dir or _default_out_dir(result_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = [
        fig_latency_by_topology(data, out_dir),
        fig_placement_speedup(data, out_dir),
        fig_phase_breakdown(data, out_dir),
        fig_workload_setup(data, out_dir),
    ]
    print(f"Wrote {len(written)} MoE figures to {out_dir}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
