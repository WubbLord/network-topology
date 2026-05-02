#!/usr/bin/env python3
"""Generate figures for the square-left matmul sweep."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from matplotlib.colors import LogNorm

import plot_results


TOPOLOGY_ORDER = [
    "Circulant {1,5,17}",
    "Torus 4x4x4",
    "Mesh 4x4x4",
    "Ring 64",
]
TOPOLOGY_LABELS = {
    "Circulant {1,5,17}": "Circulant\n{1,5,17}",
    "Torus 4x4x4": "Torus\n4x4x4",
    "Mesh 4x4x4": "Mesh\n4x4x4",
    "Ring 64": "Ring\n64",
}
TOPOLOGY_COLORS = {
    "Circulant {1,5,17}": "#6F4EAE",
    "Torus 4x4x4": "#3D8CCB",
    "Mesh 4x4x4": "#D65F5F",
    "Ring 64": "#D9A441",
}
N_ORDER = [4096, 16384, 65536, 262144]
T_ORDER = [2048, 4096]
TEXT_COLOR = "#222222"
GRID_COLOR = "#E6E6E6"


def _dim_label(value: int) -> str:
    if value >= 1024 and value % 1024 == 0:
        return f"{value // 1024}K"
    return str(value)


def _shape_label(n_size: int, t_size: int) -> str:
    return f"{_dim_label(n_size)}x{_dim_label(t_size)}"


def _workload_label(n_size: int, t_size: int) -> str:
    return f"Square {_shape_label(n_size, t_size)}"


def _duration_label(seconds: float) -> str:
    if seconds == 0:
        return "0"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f}us"
    if seconds < 1:
        return f"{seconds * 1e3:.1f}ms"
    if seconds < 60:
        return f"{seconds:.2g}s"
    if seconds < 3600:
        return f"{seconds / 60:.2g}m"
    return f"{seconds / 3600:.2g}h"


def _bytes_label(value: float) -> str:
    gib = 1024**3
    mib = 1024**2
    if value >= gib:
        return f"{value / gib:.2g} GiB"
    return f"{value / mib:.0f} MiB"


def _text_color(value: float, norm: LogNorm, cmap) -> str:
    rgba = cmap(float(norm(value)))
    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "black" if luminance > 0.58 else "white"


def _parse_desc(desc: str, params: dict) -> tuple[int, int]:
    n_size = int(params.get("N", 0))
    t_size = int(params.get("T", 0))
    if n_size and t_size:
        return n_size, t_size

    match = re.match(r"^Square (\d+)Kx(\d+)K$", desc)
    if not match:
        raise ValueError(f"Unexpected square workload name: {desc}")
    n_k, t_k = map(int, match.groups())
    return n_k * 1024, t_k * 1024


def _result_groups(run_dir: Path) -> list[dict]:
    aggregate = run_dir / "results.json"
    if aggregate.exists():
        return [json.loads(aggregate.read_text(encoding="utf-8"))]
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(run_dir.glob("*.json"))
        if path.name not in {"results.json", "milestone2_analysis.json"}
    ]


def load_rows(run_dir: Path) -> list[dict]:
    rows_by_key = {}
    for data in _result_groups(run_dir):
        map_chips = int(data.get("map_chips", 64))
        for result in data.get("results", []):
            try:
                n_size, t_size = _parse_desc(
                    result.get("desc", ""), result.get("params", {})
                )
            except ValueError:
                continue
            for topology, topo_result in result.get("topologies", {}).items():
                transfers = topo_result.get("per_transfer", [])
                serialization = sum(
                    float(t.get("serialization_latency", 0.0)) for t in transfers
                )
                first_byte = sum(
                    float(t.get("first_byte_latency", 0.0)) for t in transfers
                )
                max_link_load = max(
                    (float(t.get("max_link_load", 0.0)) for t in transfers),
                    default=0.0,
                )
                dominant = max(
                    transfers,
                    key=lambda t: float(t.get("serialization_latency", 0.0)),
                    default={},
                )
                key = (map_chips, n_size, t_size, topology)
                rows_by_key[key] = {
                    "map_chips": map_chips,
                    "n_size": n_size,
                    "t_size": t_size,
                    "workload_label": _workload_label(n_size, t_size),
                    "topology": topology,
                    "total_latency": float(topo_result.get("total_latency", 0.0)),
                    "total_energy": float(topo_result.get("total_energy", 0.0)),
                    "compute_latency": float(topo_result.get("compute_latency", 0.0)),
                    "compute_energy": float(topo_result.get("compute_energy", 0.0)),
                    "total_network_bytes": float(
                        topo_result.get("total_network_bytes", 0.0)
                    ),
                    "serialization_latency": serialization,
                    "first_byte_latency": first_byte,
                    "max_link_load": max_link_load,
                    "transfer_count": len(transfers),
                    "dominant_collective": dominant.get("collective", ""),
                    "dominant_tensor": dominant.get("tensor", ""),
                    "feedback_converged": bool(topo_result.get("feedback_converged")),
                    "feedback_iteration_count": len(
                        topo_result.get("feedback_iterations", [])
                    ),
                }
    return list(rows_by_key.values())


def _standard_entries(rows: list[dict]) -> dict:
    entries = {}
    for row in rows:
        desc = row["workload_label"]
        workload_result = {
            "desc": desc,
            "params": {
                "N": row["n_size"],
                "T": row["t_size"],
                "M": row["n_size"],
                "KN": row["t_size"],
            },
        }
        topology_result = {
            "compute_energy": row["compute_energy"],
            "compute_latency": row["compute_latency"],
            "total_energy": row["total_energy"],
            "total_latency": row["total_latency"],
            "total_network_bytes": row["total_network_bytes"],
        }
        entries[(desc, row["topology"])] = (workload_result, topology_result)
    return entries


def plot_workloads(rows: list[dict], out_dir: Path) -> Path:
    shapes = _ordered_shapes(rows)
    ncols = 4
    nrows = (len(shapes) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.0 * nrows), squeeze=False)
    fig.patch.set_facecolor("white")
    palette = ["#D8C7A3", "#B6D7A8", "#A4C2F4", "#D5A6BD", "#F4A7A0", "#F9D986"]
    for ax, (n_size, t_size), color in zip(axes.flat, shapes, palette * 4):
        a_bytes = n_size * n_size
        b_bytes = n_size * t_size
        c_bytes = n_size * t_size
        ax.set_facecolor("#FAFAFA")
        ax.barh(["A0 N x N", "B0 N x T", "C0 N x T"], [a_bytes, b_bytes, c_bytes], color=color)
        ax.set_title(_workload_label(n_size, t_size), fontsize=11, fontweight="bold")
        ax.set_xlabel("bytes, 8-bit values")
        ax.grid(axis="x", color=GRID_COLOR, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for i, value in enumerate([a_bytes, b_bytes, c_bytes]):
            ax.text(value * 1.01, i, _bytes_label(value), va="center", fontsize=8)
    for ax in axes.flat[len(shapes):]:
        ax.axis("off")
    fig.suptitle("Square Workloads", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = out_dir / "1_workloads.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def generate_standard_figures(rows: list[dict], out_dir: Path) -> None:
    entries = _standard_entries(rows)
    if not entries:
        return

    old_order = plot_results.TOPO_ORDER
    old_short = plot_results.TOPO_SHORT
    try:
        plot_results.TOPO_ORDER = list(TOPOLOGY_ORDER)
        plot_results.TOPO_SHORT = {
            **old_short,
            **{
                "Circulant {1,5,17}": "Circulant\n{1,5,17}",
                "Torus 4x4x4": "Torus\n4x4x4",
                "Mesh 4x4x4": "Mesh\n4x4x4",
                "Ring 64": "Ring\n64",
            },
        }
        plot_workloads(rows, out_dir)
        plot_results.fig_topologies(out_dir)
        plot_results.fig_latency_comparison(entries, out_dir)
        plot_results.fig_energy_breakdown(entries, out_dir)
        plot_results.fig_latency_breakdown(entries, out_dir)
        plot_results.fig_torus_aspect(entries, out_dir)
        plot_results.fig_gpt3_stress(out_dir)
        plot_results.fig_summary(entries, out_dir)
    finally:
        plot_results.TOPO_ORDER = old_order
        plot_results.TOPO_SHORT = old_short


def _ordered_shapes(rows: list[dict]) -> list[tuple[int, int]]:
    present = {(r["n_size"], r["t_size"]) for r in rows}
    ordered = []
    for n_size in N_ORDER:
        for t_size in T_ORDER:
            key = (n_size, t_size)
            if key in present:
                ordered.append(key)
    return ordered


def _matrix(rows: list[dict], metric: str):
    shapes = _ordered_shapes(rows)
    topologies = [t for t in TOPOLOGY_ORDER if any(r["topology"] == t for r in rows)]
    matrix = np.full((len(shapes), len(topologies)), np.nan)
    index = {(r["n_size"], r["t_size"], r["topology"]): r for r in rows}
    labels = []
    for i, (n_size, t_size) in enumerate(shapes):
        labels.append(_workload_label(n_size, t_size))
        for j, topology in enumerate(topologies):
            row = index.get((n_size, t_size, topology))
            if row:
                matrix[i, j] = row[metric]
    return matrix, labels, topologies


def _heatmap(rows: list[dict], out_dir: Path, metric: str, title: str, filename: str):
    matrix, labels, topologies = _matrix(rows, metric)
    values = matrix[np.isfinite(matrix) & (matrix > 0)]
    vmin = float(values.min())
    vmax = float(values.max())
    if vmin == vmax:
        vmax = vmin * 1.01
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("YlGnBu")

    fig, ax = plt.subplots(figsize=(8.5, max(5.5, 0.52 * len(labels) + 2)))
    fig.patch.set_facecolor("white")
    image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xticks(range(len(topologies)))
    ax.set_xticklabels([TOPOLOGY_LABELS.get(t, t) for t in topologies], fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                label = _duration_label(value) if "latency" in metric else f"{value:.2e}"
                ax.text(
                    j,
                    i,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    fontweight="bold",
                    color=_text_color(float(value), norm, cmap),
                )
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("seconds, log scale" if "latency" in metric else "joules, log scale")
    fig.tight_layout()
    path = out_dir / filename
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_network_energy_comparison(rows: list[dict], out_dir: Path) -> Path:
    shapes = _ordered_shapes(rows)
    topologies = [t for t in TOPOLOGY_ORDER if any(r["topology"] == t for r in rows)]
    lookup = {(r["n_size"], r["t_size"], r["topology"]): r for r in rows}

    fig, ax = plt.subplots(figsize=(max(12, 1.9 * len(shapes)), 6))
    fig.patch.set_facecolor("white")
    x = np.arange(len(shapes))
    width = min(0.15, 0.8 / len(topologies))
    offsets = np.arange(len(topologies)) - (len(topologies) - 1) / 2

    for i, topology in enumerate(topologies):
        values = [
            lookup.get((n_size, t_size, topology), {}).get("total_energy", 0.0)
            for n_size, t_size in shapes
        ]
        ax.bar(
            x + offsets[i] * width,
            values,
            width * 0.9,
            label=topology,
            color=TOPOLOGY_COLORS.get(topology, "#CCCCCC"),
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
        )

    ax.set_facecolor("#FAFAFA")
    ax.set_title(
        "Network Energy by Workload and Topology",
        fontsize=13,
        fontweight="bold",
        color=TEXT_COLOR,
        pad=12,
    )
    ax.set_ylabel("Network Energy (J)", fontsize=11, color=TEXT_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels([_shape_label(n_size, t_size) for n_size, t_size in shapes])
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(colors=TEXT_COLOR, labelsize=10)
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#DDDDDD",
        fontsize=10,
        loc="upper left",
        ncol=min(len(topologies), 4),
    )
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:,.2g}")
    fig.tight_layout()
    path = out_dir / "12_network_energy_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_report_latency_comparison(rows: list[dict], report_dir: Path) -> Path:
    shapes = _ordered_shapes(rows)
    topologies = [t for t in TOPOLOGY_ORDER if any(r["topology"] == t for r in rows)]
    lookup = {(r["n_size"], r["t_size"], r["topology"]): r for r in rows}

    fig, ax = plt.subplots(figsize=(14.2, 6.0))
    fig.patch.set_facecolor("white")
    x = np.arange(len(shapes))
    width = min(0.15, 0.8 / len(topologies))
    offsets = np.arange(len(topologies)) - (len(topologies) - 1) / 2
    plotted_values = []

    for i, topology in enumerate(topologies):
        values = [
            lookup.get((n_size, t_size, topology), {}).get("total_latency", 0.0)
            * 1e3
            for n_size, t_size in shapes
        ]
        plotted_values.extend(v for v in values if v > 0)
        ax.bar(
            x + offsets[i] * width,
            values,
            width * 0.9,
            label=topology,
            color=TOPOLOGY_COLORS.get(topology, "#CCCCCC"),
            edgecolor="white",
            linewidth=1.5,
            zorder=3,
        )

    ax.set_facecolor("#FAFAFA")
    ax.set_yscale("log")
    if plotted_values:
        ax.set_ylim(max(min(plotted_values) * 0.6, 0.2), max(plotted_values) * 1.35)
    ax.set_ylabel("Network Latency (ms)", fontsize=11, color=TEXT_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [_shape_label(n_size, t_size) for n_size, t_size in shapes],
        fontsize=10,
    )
    ax.grid(axis="y", which="major", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(colors=TEXT_COLOR, labelsize=10)
    ax.tick_params(which="minor", length=0)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda value, _: f"{value:,.0f}" if value >= 1 else f"{value:.1f}"
        )
    )
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#DDDDDD",
        fontsize=10,
        loc="upper left",
        ncol=2,
    )
    fig.tight_layout()

    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "square_latency_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def write_summary(rows: list[dict], out_dir: Path) -> tuple[Path, Path]:
    csv_path = out_dir / "square_matmul_summary.csv"
    fields = [
        "map_chips",
        "n_size",
        "t_size",
        "topology",
        "total_latency",
        "serialization_latency",
        "first_byte_latency",
        "total_energy",
        "total_network_bytes",
        "max_link_load",
        "transfer_count",
        "dominant_collective",
        "dominant_tensor",
        "feedback_converged",
        "feedback_iteration_count",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda r: (
                N_ORDER.index(r["n_size"]),
                T_ORDER.index(r["t_size"]),
                TOPOLOGY_ORDER.index(r["topology"]),
            ),
        ):
            writer.writerow({field: row[field] for field in fields})

    grouped = {}
    for row in rows:
        grouped.setdefault((row["n_size"], row["t_size"]), []).append(row)
    md_path = out_dir / "summary.md"
    lines = [
        "# Square Matmul Sweep Summary",
        "",
        f"Completed topology results: {len(rows)}",
        f"Complete four-topology workload blocks: {sum(1 for v in grouped.values() if len(v) >= 4)}",
        "",
        "| Workload | Best topology | Latency | Max-link term | Iterations | Worst/best |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key in sorted(grouped, key=lambda k: (N_ORDER.index(k[0]), T_ORDER.index(k[1]))):
        entries = grouped[key]
        best = min(entries, key=lambda r: r["total_latency"])
        worst = max(entries, key=lambda r: r["total_latency"])
        speedup = worst["total_latency"] / best["total_latency"] if best["total_latency"] else 0.0
        lines.append(
            f"| {_workload_label(*key)} | {best['topology']} | "
            f"{_duration_label(best['total_latency'])} | "
            f"{_duration_label(best['serialization_latency'])} | "
            f"{best['feedback_iteration_count']} | {speedup:.2f}x |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def reset_generated_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file() and child.suffix in {".png", ".csv", ".md"}:
            child.unlink()


def mirror_outputs(out_dir: Path, mirror_dir: Path | None) -> None:
    if mirror_dir is None:
        return
    if out_dir.resolve() != mirror_dir.resolve():
        reset_generated_dir(mirror_dir)
    for path in out_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, mirror_dir / path.name)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot square matmul sweep results.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--mirror-dir", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.run_dir)
    if not rows:
        raise SystemExit(f"No square matmul results found under {args.run_dir}")
    out_dir = args.out_dir or args.run_dir / "figures"
    reset_generated_dir(out_dir)

    generate_standard_figures(rows, out_dir)
    paths = [
        _heatmap(rows, out_dir, "total_latency", "Square Matmul Network Latency", "9_network_latency.png"),
        _heatmap(rows, out_dir, "serialization_latency", "Square Matmul Max-Link Serialization", "10_max_link_serialization.png"),
        _heatmap(rows, out_dir, "total_energy", "Square Matmul Network Energy", "11_network_energy.png"),
        plot_network_energy_comparison(rows, out_dir),
    ]
    paths.extend(write_summary(rows, out_dir))
    if args.report_dir is not None:
        paths.append(plot_report_latency_comparison(rows, args.report_dir))
    mirror_outputs(out_dir, args.mirror_dir)

    generated = sorted(path for path in out_dir.iterdir() if path.is_file())
    print(f"Generated {len(generated)} square matmul artifacts in {out_dir}")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
