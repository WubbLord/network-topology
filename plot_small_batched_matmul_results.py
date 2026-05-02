#!/usr/bin/env python3
"""Generate figures for the focused small batched matmul sweep."""

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
BATCH_ORDER = [1, 16, 256]
SHAPE_ORDER = [(2048, 8192), (4096, 16384)]
TEXT_COLOR = "#222222"
GRID_COLOR = "#E6E6E6"


def _dim_label(value: int) -> str:
    if value >= 1024 and value % 1024 == 0:
        return f"{value // 1024}K"
    return str(value)


def _shape_label(m_size: int, kn_size: int) -> str:
    return f"{_dim_label(m_size)}x{_dim_label(kn_size)}"


def _workload_label(batch_size: int, m_size: int, kn_size: int) -> str:
    shape = _shape_label(m_size, kn_size)
    return shape if batch_size == 1 else f"B{batch_size} {shape}"


def _parse_desc(desc: str, params: dict) -> tuple[int, int, int]:
    batch_size = int(params.get("BATCH_SIZE", 1))
    m_size = int(params.get("M", 0))
    kn_size = int(params.get("KN", 0))
    if m_size and kn_size:
        return batch_size, m_size, kn_size

    match = re.search(r"(\d+)Kx(\d+)K$", desc)
    if not match:
        raise ValueError(f"Unexpected workload name: {desc}")
    m_k, kn_k = map(int, match.groups())
    return batch_size, m_k * 1024, kn_k * 1024


def _duration_label(seconds: float) -> str:
    if seconds == 0:
        return "0"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f}us"
    if seconds < 1:
        return f"{seconds * 1e3:.1f}ms"
    if seconds < 60:
        return f"{seconds:.2g}s"
    return f"{seconds / 60:.2g}m"


def _text_color(value: float, norm: LogNorm, cmap) -> str:
    rgba = cmap(float(norm(value)))
    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "black" if luminance > 0.58 else "white"


def load_rows(run_dir: Path) -> list[dict]:
    data_path = run_dir / "results.json"
    if data_path.exists():
        data = json.loads(data_path.read_text(encoding="utf-8"))
        result_groups = [data]
    else:
        result_groups = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(run_dir.glob("*.json"))
            if path.name not in {"results.json", "milestone2_analysis.json"}
        ]

    rows_by_key = {}
    for data in result_groups:
        map_chips = int(data.get("map_chips", 64))
        for result in data.get("results", []):
            try:
                batch_size, m_size, kn_size = _parse_desc(
                    result.get("desc", ""), result.get("params", {})
                )
            except ValueError:
                continue
            label = _workload_label(batch_size, m_size, kn_size)
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
                key = (batch_size, m_size, kn_size, topology)
                rows_by_key[key] = {
                    "map_chips": map_chips,
                    "batch_size": batch_size,
                    "m_size": m_size,
                    "kn_size": kn_size,
                    "workload_label": label,
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
                "N_EINSUMS": 1,
                "M": row["m_size"],
                "KN": row["kn_size"],
                **(
                    {"BATCH_SIZE": row["batch_size"]}
                    if row["batch_size"] != 1
                    else {}
                ),
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


def generate_standard_figures(rows: list[dict], out_dir: Path) -> None:
    entries = _standard_entries(rows)
    if not entries:
        return

    old_order = plot_results.TOPO_ORDER
    old_short = plot_results.TOPO_SHORT
    try:
        old_show_titles = getattr(plot_results, "SHOW_FIGURE_TITLES", True)
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
        plot_results.SHOW_FIGURE_TITLES = False
        plot_results.fig_workloads(out_dir, entries)
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
        plot_results.SHOW_FIGURE_TITLES = old_show_titles


def _ordered_workloads(rows: list[dict]) -> list[tuple[int, int, int]]:
    present = {(r["batch_size"], r["m_size"], r["kn_size"]) for r in rows}
    ordered = []
    for batch_size in BATCH_ORDER:
        for m_size, kn_size in SHAPE_ORDER:
            key = (batch_size, m_size, kn_size)
            if key in present:
                ordered.append(key)
    return ordered


def plot_max_link_heatmap(rows: list[dict], out_dir: Path) -> Path:
    workloads = _ordered_workloads(rows)
    topologies = [t for t in TOPOLOGY_ORDER if any(r["topology"] == t for r in rows)]
    matrix = np.full((len(workloads), len(topologies)), np.nan)
    index = {
        (r["batch_size"], r["m_size"], r["kn_size"], r["topology"]): r for r in rows
    }
    labels = []
    for i, (batch_size, m_size, kn_size) in enumerate(workloads):
        labels.append(_workload_label(batch_size, m_size, kn_size))
        for j, topology in enumerate(topologies):
            row = index.get((batch_size, m_size, kn_size, topology))
            if row:
                matrix[i, j] = row["serialization_latency"]

    values = matrix[np.isfinite(matrix) & (matrix > 0)]
    vmin = float(values.min())
    vmax = float(values.max())
    if vmin == vmax:
        vmax = vmin * 1.01
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("YlGnBu")

    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.5 * len(labels) + 2)))
    fig.patch.set_facecolor("white")
    image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
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
                ax.text(
                    j,
                    i,
                    _duration_label(float(value)),
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    fontweight="bold",
                    color=_text_color(float(value), norm, cmap),
                )
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("seconds, log scale")
    fig.tight_layout()
    path = out_dir / "9_max_link_serialization.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_batch_scaling(rows: list[dict], out_dir: Path) -> Path:
    topologies = [t for t in TOPOLOGY_ORDER if any(r["topology"] == t for r in rows)]
    lookup = {
        (r["batch_size"], r["m_size"], r["kn_size"], r["topology"]): r for r in rows
    }
    fig, axes = plt.subplots(1, len(SHAPE_ORDER), figsize=(11, 4.5), squeeze=False)
    fig.patch.set_facecolor("white")
    for ax, (m_size, kn_size) in zip(axes.flat, SHAPE_ORDER):
        for topology in topologies:
            y = [
                lookup.get((batch_size, m_size, kn_size, topology), {}).get(
                    "serialization_latency", np.nan
                )
                for batch_size in BATCH_ORDER
            ]
            ax.plot(
                BATCH_ORDER,
                y,
                marker="o",
                linewidth=2,
                color=TOPOLOGY_COLORS.get(topology),
                label=topology,
            )
        ax.set_title(_shape_label(m_size, kn_size), fontsize=12, fontweight="bold")
        ax.set_yscale("log")
        ax.set_xticks(BATCH_ORDER)
        ax.set_xlabel("batch size")
        ax.set_ylabel("seconds")
        ax.grid(True, which="both", color=GRID_COLOR, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, loc="lower center")
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    path = out_dir / "10_batch_scaling_max_link_serialization.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_network_energy_comparison_by_batch(rows: list[dict], out_dir: Path) -> list[Path]:
    paths = []
    topologies = [t for t in TOPOLOGY_ORDER if any(r["topology"] == t for r in rows)]
    for batch_size in BATCH_ORDER:
        batch_rows = [row for row in rows if row["batch_size"] == batch_size]
        if not batch_rows:
            continue

        workloads = [
            (m_size, kn_size)
            for m_size, kn_size in SHAPE_ORDER
            if any(row["m_size"] == m_size and row["kn_size"] == kn_size for row in batch_rows)
        ]
        if not workloads or len(topologies) < 2:
            continue

        lookup = {
            (row["m_size"], row["kn_size"], row["topology"]): row
            for row in batch_rows
        }
        fig, ax = plt.subplots(figsize=(max(9, 2.1 * len(workloads)), 5.5))
        fig.patch.set_facecolor("white")
        x = np.arange(len(workloads))
        width = min(0.15, 0.8 / len(topologies))
        offsets = np.arange(len(topologies)) - (len(topologies) - 1) / 2

        for i, topology in enumerate(topologies):
            values = [
                lookup.get((m_size, kn_size, topology), {}).get("total_energy", 0.0)
                for m_size, kn_size in workloads
            ]
            ax.bar(
                x + offsets[i] * width,
                values,
                width * 0.9,
                label=topology,
                color=TOPOLOGY_COLORS.get(topology, "#CCCCCC"),
                edgecolor="white",
                linewidth=1.4,
                zorder=3,
            )

        ax.set_facecolor("#FAFAFA")
        ax.set_ylabel("Network Energy (J)", fontsize=11, color=TEXT_COLOR)
        ax.set_xticks(x)
        ax.set_xticklabels([_shape_label(m_size, kn_size) for m_size, kn_size in workloads])
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
        path = out_dir / f"11_network_energy_comparison_B{batch_size}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        paths.append(path)
    return paths


def write_summary(rows: list[dict], out_dir: Path) -> tuple[Path, Path]:
    csv_path = out_dir / "small_batched_matmul_summary.csv"
    fields = [
        "map_chips",
        "batch_size",
        "m_size",
        "kn_size",
        "topology",
        "total_latency",
        "serialization_latency",
        "first_byte_latency",
        "total_energy",
        "total_network_bytes",
        "max_link_load",
        "transfer_count",
        "feedback_converged",
        "feedback_iteration_count",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda r: (
                BATCH_ORDER.index(r["batch_size"]),
                SHAPE_ORDER.index((r["m_size"], r["kn_size"])),
                TOPOLOGY_ORDER.index(r["topology"]),
            ),
        ):
            writer.writerow({field: row[field] for field in fields})

    grouped = {}
    for row in rows:
        key = (row["batch_size"], row["m_size"], row["kn_size"])
        grouped.setdefault(key, []).append(row)
    md_path = out_dir / "summary.md"
    lines = [
        "# Small Batched Matmul Summary",
        "",
        f"Completed topology results: {len(rows)}",
        f"Complete four-topology workload blocks: {sum(1 for v in grouped.values() if len(v) >= 4)}",
        "",
        "| Workload | Best topology | Latency | Max-link term | Iterations | Worst/best |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key in sorted(
        grouped,
        key=lambda k: (BATCH_ORDER.index(k[0]), SHAPE_ORDER.index((k[1], k[2]))),
    ):
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
    parser = argparse.ArgumentParser(description="Plot small batched matmul results.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--mirror-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.run_dir)
    if not rows:
        raise SystemExit(f"No small batched matmul results found under {args.run_dir}")
    out_dir = args.out_dir or args.run_dir / "figures"
    reset_generated_dir(out_dir)

    generate_standard_figures(rows, out_dir)
    paths = [
        plot_max_link_heatmap(rows, out_dir),
        plot_batch_scaling(rows, out_dir),
    ]
    paths.extend(plot_network_energy_comparison_by_batch(rows, out_dir))
    paths.extend(write_summary(rows, out_dir))
    mirror_outputs(out_dir, args.mirror_dir)

    generated = sorted(path for path in out_dir.iterdir() if path.is_file())
    print(f"Generated {len(generated)} small batched matmul artifacts in {out_dir}")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
