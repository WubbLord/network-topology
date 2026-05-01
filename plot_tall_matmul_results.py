#!/usr/bin/env python3
"""Generate figures for the tall matmul MAP_CHIPS sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


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
N_ORDER = [4096, 16384, 65536, 262144]
K_ORDER = [2048, 4096]
TEXT_COLOR = "#222222"
GRID_COLOR = "#E6E6E6"


def _dim_label(value: int) -> str:
    if value >= 1024 and value % 1024 == 0:
        return f"{value // 1024}K"
    return str(value)


def _shape_label(n_size: int, k_size: int) -> str:
    return f"{_dim_label(n_size)}x{_dim_label(k_size)}"


def _parse_workload(name: str) -> tuple[int, int, int]:
    match = re.match(r"^Tall B(\d+) N(\d+)Kx(\d+)K$", name)
    if not match:
        raise ValueError(f"Unexpected tall workload name: {name}")
    batch, n_k, k_k = map(int, match.groups())
    return batch, n_k * 1024, k_k * 1024


def _workload_label(batch: int, n_size: int, k_size: int) -> str:
    if batch == 1:
        return _shape_label(n_size, k_size)
    return f"B{batch} {_shape_label(n_size, k_size)}"


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


def _text_color(value: float, norm: LogNorm, cmap) -> str:
    rgba = cmap(float(norm(value)))
    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "black" if luminance > 0.58 else "white"


def _result_json_paths(map_dir: Path) -> list[Path]:
    aggregate = map_dir / "results.json"
    if aggregate.exists():
        return [aggregate]
    return sorted(p for p in map_dir.glob("*.json") if p.name != "results.json")


def load_rows(run_root: Path) -> list[dict]:
    rows_by_key = {}
    for map_dir in sorted(run_root.glob("map*")):
        for result_path in _result_json_paths(map_dir):
            data = json.loads(result_path.read_text(encoding="utf-8"))
            map_chips = int(data.get("map_chips", map_dir.name.replace("map", "")))
            for result in data.get("results", []):
                try:
                    batch_size, n_size, k_size = _parse_workload(result["desc"])
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
                    key = (map_chips, batch_size, n_size, k_size, topology)
                    rows_by_key[key] = {
                        "map_chips": map_chips,
                        "batch_size": batch_size,
                        "n_size": n_size,
                        "k_size": k_size,
                        "workload": result["desc"],
                        "workload_label": _workload_label(batch_size, n_size, k_size),
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
                    }
    return list(rows_by_key.values())


def _standard_plot_entries(rows: list[dict]) -> dict:
    entries = {}
    for row in rows:
        desc = row["workload_label"]
        params = {
            "N_EINSUMS": 1,
            "M": row["n_size"],
            "KN": row["k_size"],
        }
        if row["batch_size"] != 1:
            params["BATCH_SIZE"] = row["batch_size"]
        workload_result = {
            "desc": desc,
            "params": params,
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
    entries = _standard_plot_entries(rows)
    if not entries:
        return

    from plot_results import (
        fig_energy_breakdown,
        fig_gpt3_stress,
        fig_latency_breakdown,
        fig_summary,
        fig_topologies,
        fig_torus_aspect,
        fig_workloads,
    )

    fig_workloads(out_dir, entries)
    fig_topologies(out_dir)
    plot_latency_comparison_by_batch(rows, out_dir)
    plot_standard_figure_by_batch(
        rows,
        out_dir,
        fig_energy_breakdown,
        "4_energy_breakdown.png",
        "4_energy_breakdown_B{batch}.png",
    )
    plot_standard_figure_by_batch(
        rows,
        out_dir,
        fig_latency_breakdown,
        "5_latency_breakdown.png",
        "5_latency_breakdown_B{batch}.png",
    )
    fig_torus_aspect(entries, out_dir)
    fig_gpt3_stress(out_dir)
    fig_summary(entries, out_dir)


def plot_standard_figure_by_batch(
    rows: list[dict],
    out_dir: Path,
    plot_fn,
    source_name: str,
    output_template: str,
) -> list[Path]:
    paths = []
    source = out_dir / source_name
    for batch in BATCH_ORDER:
        batch_rows = [row for row in rows if row["batch_size"] == batch]
        entries = _standard_plot_entries(batch_rows)
        if not entries:
            continue

        if source.exists():
            source.unlink()
        plot_fn(entries, out_dir)
        if not source.exists():
            continue

        path = out_dir / output_template.format(batch=batch)
        if path.exists():
            path.unlink()
        shutil.move(str(source), str(path))
        paths.append(path)
    return paths


def plot_latency_comparison_by_batch(rows: list[dict], out_dir: Path) -> list[Path]:
    paths = []
    for batch in BATCH_ORDER:
        batch_rows = [row for row in rows if row["batch_size"] == batch]
        if not batch_rows:
            continue

        workloads = [
            (batch, n_size, k_size)
            for n_size in N_ORDER
            for k_size in K_ORDER
            if any(row["n_size"] == n_size and row["k_size"] == k_size for row in batch_rows)
        ]
        topologies = [
            topology
            for topology in TOPOLOGY_ORDER
            if any(row["topology"] == topology for row in batch_rows)
        ]
        if not workloads or len(topologies) < 2:
            continue

        fig, ax = plt.subplots(figsize=(max(12, 1.9 * len(workloads)), 6))
        fig.patch.set_facecolor("white")
        x = np.arange(len(workloads))
        width = min(0.15, 0.8 / len(topologies))
        offsets = np.arange(len(topologies)) - (len(topologies) - 1) / 2
        lookup = {
            (row["n_size"], row["k_size"], row["topology"]): row
            for row in batch_rows
        }

        for i, topology in enumerate(topologies):
            values = [
                lookup.get((n_size, k_size, topology), {}).get("total_latency", 0.0)
                * 1e3
                for _, n_size, k_size in workloads
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

        batch_label = "B1" if batch == 1 else f"B{batch}"
        ax.set_facecolor("#FAFAFA")
        ax.set_title(
            f"Network Latency by Workload and Topology ({batch_label})",
            fontsize=13,
            fontweight="bold",
            color=TEXT_COLOR,
            pad=12,
        )
        ax.set_ylabel("Network Latency (ms)", fontsize=11, color=TEXT_COLOR)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [_shape_label(n_size, k_size) for _, n_size, k_size in workloads],
            fontsize=10,
        )
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
        ax.yaxis.set_major_formatter(lambda value, _: f"{value:,.0f}")
        fig.tight_layout()

        path = out_dir / f"3_latency_comparison_B{batch}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  Saved {path}")
        paths.append(path)
    return paths


def _ordered_workloads(rows: list[dict]) -> list[tuple[int, int, int]]:
    present = {(r["batch_size"], r["n_size"], r["k_size"]) for r in rows}
    ordered = []
    for batch in BATCH_ORDER:
        for n_size in N_ORDER:
            for k_size in K_ORDER:
                key = (batch, n_size, k_size)
                if key in present:
                    ordered.append(key)
    return ordered


def _matrix(rows: list[dict], map_chips: int, metric: str):
    workloads = _ordered_workloads([r for r in rows if r["map_chips"] == map_chips])
    topologies = [t for t in TOPOLOGY_ORDER if any(r["topology"] == t for r in rows)]
    matrix = np.full((len(workloads), len(topologies)), np.nan)
    labels = []
    index = {
        (r["batch_size"], r["n_size"], r["k_size"], r["topology"]): r
        for r in rows
        if r["map_chips"] == map_chips
    }
    for i, (batch, n_size, k_size) in enumerate(workloads):
        labels.append(_workload_label(batch, n_size, k_size))
        for j, topology in enumerate(topologies):
            row = index.get((batch, n_size, k_size, topology))
            if row:
                matrix[i, j] = row[metric]
    return matrix, labels, topologies


def _heatmap(rows: list[dict], out_dir: Path, metric: str, title: str, filename: str):
    map_values = sorted({r["map_chips"] for r in rows})
    matrices = [_matrix(rows, map_chips, metric) for map_chips in map_values]
    values = np.concatenate(
        [m[np.isfinite(m) & (m > 0)] for m, _, _ in matrices if np.isfinite(m).any()]
    )
    vmin = float(values.min())
    vmax = float(values.max())
    if vmin == vmax:
        vmax = vmin * 1.01
    norm = LogNorm(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("YlGnBu")
    fig, axes = plt.subplots(
        1,
        len(map_values),
        figsize=(7.2 * len(map_values), 10.5),
        sharey=True,
        squeeze=False,
    )
    fig.patch.set_facecolor("white")
    image = None
    for ax, map_chips, (matrix, labels, topologies) in zip(
        axes.flat, map_values, matrices
    ):
        image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(len(topologies)))
        ax.set_xticklabels([TOPOLOGY_LABELS.get(t, t) for t in topologies], fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
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
                        _duration_label(value) if "latency" in metric else f"{value:.2e}",
                        ha="center",
                        va="center",
                        fontsize=6.6,
                        fontweight="bold",
                        color=_text_color(float(value), norm, cmap),
                    )
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.12, right=0.88, bottom=0.06, top=0.92, wspace=0.04)
    cax = fig.add_axes([0.90, 0.18, 0.018, 0.62])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("seconds, log scale" if "latency" in metric else "joules, log scale")
    path = out_dir / filename
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_batch_scaling(
    rows: list[dict], out_dir: Path, title: str, filename: str
) -> Path:
    map_values = sorted({r["map_chips"] for r in rows})
    topologies = [t for t in TOPOLOGY_ORDER if any(r["topology"] == t for r in rows)]
    present_shapes = {(r["map_chips"], r["n_size"], r["k_size"]) for r in rows}
    panels = [
        (m, n, k)
        for m in map_values
        for n in N_ORDER
        for k in K_ORDER
        if (m, n, k) in present_shapes
    ]
    ncols = 4
    nrows = math.ceil(len(panels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(17, 3.0 * nrows), squeeze=False)
    fig.patch.set_facecolor("white")
    lookup = {
        (r["map_chips"], r["n_size"], r["k_size"], r["batch_size"], r["topology"]): r
        for r in rows
    }
    for ax, (map_chips, n_size, k_size) in zip(axes.flat, panels):
        for topology in topologies:
            y = [
                lookup.get((map_chips, n_size, k_size, batch, topology), {}).get(
                    "serialization_latency", np.nan
                )
                for batch in BATCH_ORDER
            ]
            ax.plot(
                BATCH_ORDER,
                y,
                marker="o",
                linewidth=2,
                color=TOPOLOGY_COLORS.get(topology),
                label=topology,
            )
        ax.set_title(
            _shape_label(n_size, k_size),
            fontsize=10,
            fontweight="bold",
        )
        ax.set_yscale("log")
        ax.set_xticks(BATCH_ORDER)
        ax.set_xlabel("batch setting")
        ax.set_ylabel("seconds")
        ax.grid(True, which="both", color=GRID_COLOR, linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, frameon=False, loc="lower center")
    fig.suptitle(title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    path = out_dir / filename
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def write_summary(rows: list[dict], out_dir: Path) -> tuple[Path, Path]:
    csv_path = out_dir / "tall_matmul_summary.csv"
    fields = [
        "map_chips",
        "batch_size",
        "n_size",
        "k_size",
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
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda r: (
                r["map_chips"],
                BATCH_ORDER.index(r["batch_size"]),
                N_ORDER.index(r["n_size"]),
                K_ORDER.index(r["k_size"]),
                TOPOLOGY_ORDER.index(r["topology"]),
            ),
        ):
            writer.writerow({field: row[field] for field in fields})

    md_path = out_dir / "summary.md"
    grouped = {}
    for row in rows:
        key = (row["map_chips"], row["batch_size"], row["n_size"], row["k_size"])
        grouped.setdefault(key, []).append(row)
    complete_blocks = sum(
        1
        for entries in grouped.values()
        if {entry["topology"] for entry in entries} >= set(TOPOLOGY_ORDER)
    )
    lines = [
        "# Tall Matmul Sweep Summary",
        "",
        f"Completed topology results: {len(rows)}",
        f"Observed workload/MAP cases: {len(grouped)}",
        f"Complete four-topology blocks: {complete_blocks}",
        "",
        "| MAP | Workload | Best topology | Latency | Max-link term | Worst/best |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for key in sorted(
        grouped,
        key=lambda k: (k[0], BATCH_ORDER.index(k[1]), N_ORDER.index(k[2]), K_ORDER.index(k[3])),
    ):
        entries = grouped[key]
        best = min(entries, key=lambda r: r["total_latency"])
        worst = max(entries, key=lambda r: r["total_latency"])
        speedup = worst["total_latency"] / best["total_latency"] if best["total_latency"] else 0.0
        map_chips, batch, n_size, k_size = key
        lines.append(
            f"| {map_chips} | {_workload_label(batch, n_size, k_size)} | "
            f"{best['topology']} | {_duration_label(best['total_latency'])} | "
            f"{_duration_label(best['serialization_latency'])} | {speedup:.2f}x |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def reset_generated_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir() and re.fullmatch(r"map\d+", child.name):
            shutil.rmtree(child)
        elif child.is_file() and child.suffix in {".png", ".csv", ".md"}:
            child.unlink()


def mirror_outputs(out_dir: Path, mirror_dir: Path | None) -> None:
    if mirror_dir is None:
        return
    if out_dir.resolve() != mirror_dir.resolve():
        reset_generated_dir(mirror_dir)
    for path in out_dir.iterdir():
        target = mirror_dir / path.name
        if path.is_dir():
            shutil.copytree(path, target, dirs_exist_ok=True)
        elif path.is_file():
            shutil.copy2(path, mirror_dir / path.name)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot tall matmul sweep results.")
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--mirror-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.run_root)
    if not rows:
        raise SystemExit(f"No tall matmul results found under {args.run_root}")
    out_dir = args.out_dir or args.run_root / "figures"
    reset_generated_dir(out_dir)

    paths = []
    for map_chips in sorted({row["map_chips"] for row in rows}):
        map_rows = [row for row in rows if row["map_chips"] == map_chips]
        map_dir = out_dir / f"map{map_chips}"
        map_dir.mkdir(parents=True, exist_ok=True)
        generate_standard_figures(map_rows, map_dir)
        paths.extend(
            [
                _heatmap(
                    map_rows,
                    map_dir,
                    "total_latency",
                    "Network Latency",
                    "9_network_latency.png",
                ),
                _heatmap(
                    map_rows,
                    map_dir,
                    "serialization_latency",
                    "Max-Link Serialization Term",
                    "10_max_link_serialization.png",
                ),
                plot_batch_scaling(
                    map_rows,
                    map_dir,
                    "Batch Scaling of Max-Link Serialization",
                    "11_batch_scaling_max_link_serialization.png",
                ),
                _heatmap(
                    map_rows,
                    map_dir,
                    "total_energy",
                    "Network Energy",
                    "12_network_energy.png",
                ),
            ]
        )
        paths.extend(write_summary(map_rows, map_dir))
    paths.extend(write_summary(rows, out_dir))
    mirror_outputs(out_dir, args.mirror_dir)

    generated_files = sorted(path for path in out_dir.rglob("*") if path.is_file())
    print(f"Generated {len(generated_files)} tall matmul artifacts in {out_dir}")
    for path in generated_files:
        print(path)


if __name__ == "__main__":
    main()
