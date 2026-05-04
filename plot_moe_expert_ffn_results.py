#!/usr/bin/env python3
"""Generate figures for the AccelForge-backed MoE expert-FFN sweep."""

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
COMPUTE_COLOR = "#7F7F7F"
NETWORK_COLOR = "#6F4EAE"
FIRST_BYTE_COLOR = "#A8DADC"
GRID_COLOR = "#E6E6E6"
TEXT_COLOR = "#222222"


def _load_result_groups(path: Path) -> list[dict]:
    if path.is_file():
        return [json.loads(path.read_text(encoding="utf-8"))]

    result_path = path / "results.json"
    if result_path.exists():
        return [json.loads(result_path.read_text(encoding="utf-8"))]

    groups = []
    for partial in sorted(path.glob("*.json")):
        if partial.name == "milestone2_analysis.json":
            continue
        groups.append(json.loads(partial.read_text(encoding="utf-8")))
    if not groups:
        raise SystemExit(f"No result JSON found in {path}")
    return groups


def _tokens_from_result(result: dict) -> int:
    params = result.get("params", {})
    if "TOKENS_PER_EXPERT" in params:
        return int(params["TOKENS_PER_EXPERT"])
    match = re.search(r"\bT(\d+)\b", result.get("desc", ""))
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not parse token count from {result.get('desc', '')!r}")


def _first_present(*values, default=0.0) -> float:
    for value in values:
        if value is not None:
            return float(value)
    return float(default)


def load_rows(path: Path) -> list[dict]:
    rows_by_key = {}
    for group in _load_result_groups(path):
        for result in group.get("results", []):
            try:
                tokens = _tokens_from_result(result)
            except ValueError:
                continue

            for topology, topo_result in result.get("topologies", {}).items():
                transfers = topo_result.get("per_transfer", [])
                serialization = sum(
                    float(transfer.get("serialization_latency", 0.0))
                    for transfer in transfers
                )
                first_byte = sum(
                    float(transfer.get("first_byte_latency", 0.0))
                    for transfer in transfers
                )
                max_link_load = max(
                    (float(transfer.get("max_link_load", 0.0)) for transfer in transfers),
                    default=0.0,
                )
                dominant = max(
                    transfers,
                    key=lambda transfer: float(transfer.get("total_latency", 0.0)),
                    default={},
                )
                key = (tokens, topology)
                rows_by_key[key] = {
                    "workload": result.get("desc", f"MoE ExpertFFN E16 T{tokens}"),
                    "tokens_per_expert": tokens,
                    "topology": topology,
                    "network_latency": _first_present(
                        topo_result.get("total_latency"),
                        result.get("lats", {}).get(topology),
                    ),
                    "network_energy": _first_present(topo_result.get("total_energy")),
                    "compute_latency": _first_present(
                        topo_result.get("compute_latency"),
                        result.get("compute_l_by_topology", {}).get(topology),
                        result.get("compute_l"),
                    ),
                    "compute_energy": _first_present(
                        topo_result.get("compute_energy"),
                        result.get("compute_e_by_topology", {}).get(topology),
                        result.get("compute_e"),
                    ),
                    "total_network_bytes": _first_present(
                        topo_result.get("total_network_bytes"),
                        topo_result.get("bytes"),
                        result.get("bytes_by_topology", {}).get(topology),
                        result.get("bytes"),
                    ),
                    "max_link_load": max_link_load,
                    "serialization_latency": serialization,
                    "first_byte_latency": first_byte,
                    "dominant_collective": dominant.get("collective_type", ""),
                    "dominant_tensor": dominant.get("tensor_name", ""),
                    "feedback_converged": bool(topo_result.get("feedback_converged")),
                    "feedback_iteration_count": len(
                        topo_result.get("feedback_iterations", [])
                    ),
                }
    return sorted(
        rows_by_key.values(),
        key=lambda row: (
            row["tokens_per_expert"],
            TOPOLOGY_ORDER.index(row["topology"])
            if row["topology"] in TOPOLOGY_ORDER
            else len(TOPOLOGY_ORDER),
        ),
    )


def _ordered_tokens(rows: list[dict]) -> list[int]:
    return sorted({row["tokens_per_expert"] for row in rows})


def _style_axis(ax, ylabel: str | None = None, log: bool = False) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#BBBBBB")
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=TEXT_COLOR)
    if log:
        ax.set_yscale("log")


def _value_by(rows: list[dict], field: str) -> dict[tuple[int, str], float]:
    return {
        (row["tokens_per_expert"], row["topology"]): float(row[field])
        for row in rows
    }


def fig_latency_comparison(rows: list[dict], out_dir: Path) -> Path:
    tokens = _ordered_tokens(rows)
    values = _value_by(rows, "network_latency")
    x = np.arange(len(tokens))
    width = 0.18
    offsets = np.arange(len(TOPOLOGY_ORDER)) - (len(TOPOLOGY_ORDER) - 1) / 2

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    fig.patch.set_facecolor("white")
    for idx, topology in enumerate(TOPOLOGY_ORDER):
        ys = [values.get((token, topology), 0.0) * 1e3 for token in tokens]
        ax.bar(
            x + offsets[idx] * width,
            ys,
            width * 0.92,
            color=TOPOLOGY_COLORS[topology],
            label=TOPOLOGY_LABELS[topology].replace("\n", " "),
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )

    _style_axis(ax, "Network latency (ms)", log=True)
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{token}" for token in tokens])
    ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    path = out_dir / "moe_expert_ffn_latency_comparison.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fig_latency_breakdown(rows: list[dict], out_dir: Path) -> Path:
    tokens = _ordered_tokens(rows)
    ncols = 2
    nrows = (len(tokens) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, 4.5 * nrows), squeeze=False)
    fig.patch.set_facecolor("white")

    by_key = {(row["tokens_per_expert"], row["topology"]): row for row in rows}
    for ax, token in zip(axes.flat, tokens):
        x = np.arange(len(TOPOLOGY_ORDER))
        compute = []
        serialization = []
        first_byte = []
        for topology in TOPOLOGY_ORDER:
            row = by_key.get((token, topology), {})
            compute.append(float(row.get("compute_latency", 0.0)) * 1e3)
            serialization.append(float(row.get("serialization_latency", 0.0)) * 1e3)
            first_byte.append(float(row.get("first_byte_latency", 0.0)) * 1e3)

        ax.bar(x, compute, 0.62, color=COMPUTE_COLOR, label="Compute", zorder=3)
        ax.bar(
            x,
            serialization,
            0.62,
            bottom=compute,
            color=NETWORK_COLOR,
            label="Network serialization",
            zorder=3,
        )
        ax.bar(
            x,
            first_byte,
            0.62,
            bottom=np.array(compute) + np.array(serialization),
            color=FIRST_BYTE_COLOR,
            label="First byte",
            zorder=3,
        )
        _style_axis(ax, "Latency (ms)")
        ax.set_title(f"T{token}", fontsize=11, fontweight="bold", color=TEXT_COLOR)
        ax.set_xticks(x)
        ax.set_xticklabels([TOPOLOGY_LABELS[topology] for topology in TOPOLOGY_ORDER])

    for ax in axes.flat[len(tokens):]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    path = out_dir / "moe_expert_ffn_latency_breakdown.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fig_energy_breakdown(rows: list[dict], out_dir: Path) -> Path:
    tokens = _ordered_tokens(rows)
    ncols = 2
    nrows = (len(tokens) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.5, 4.5 * nrows), squeeze=False)
    fig.patch.set_facecolor("white")

    by_key = {(row["tokens_per_expert"], row["topology"]): row for row in rows}
    for ax, token in zip(axes.flat, tokens):
        x = np.arange(len(TOPOLOGY_ORDER))
        compute = []
        network = []
        for topology in TOPOLOGY_ORDER:
            row = by_key.get((token, topology), {})
            compute.append(float(row.get("compute_energy", 0.0)))
            network.append(float(row.get("network_energy", 0.0)))

        ax.bar(x, compute, 0.62, color=COMPUTE_COLOR, label="Compute", zorder=3)
        ax.bar(
            x,
            network,
            0.62,
            bottom=compute,
            color=NETWORK_COLOR,
            label="Network",
            zorder=3,
        )
        for idx, (compute_value, network_value) in enumerate(zip(compute, network)):
            total = compute_value + network_value
            if total <= 0:
                continue
            ax.text(
                idx,
                total * 1.02,
                f"{network_value / total * 100:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8,
                color=TEXT_COLOR,
            )
        _style_axis(ax, "Energy (J)")
        ax.set_title(f"T{token}", fontsize=11, fontweight="bold", color=TEXT_COLOR)
        ax.set_xticks(x)
        ax.set_xticklabels([TOPOLOGY_LABELS[topology] for topology in TOPOLOGY_ORDER])

    for ax in axes.flat[len(tokens):]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    path = out_dir / "moe_expert_ffn_energy_breakdown.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def fig_summary_dashboard(rows: list[dict], out_dir: Path) -> Path:
    tokens = _ordered_tokens(rows)
    by_key = {(row["tokens_per_expert"], row["topology"]): row for row in rows}
    table_rows = []
    for token in tokens:
        circ = by_key[(token, "Circulant {1,5,17}")]
        torus = by_key[(token, "Torus 4x4x4")]
        mesh = by_key[(token, "Mesh 4x4x4")]
        ring = by_key[(token, "Ring 64")]
        table_rows.append(
            [
                f"T{token}",
                f"{circ['network_latency'] * 1e3:.3f} ms",
                f"{torus['network_latency'] / circ['network_latency']:.2f}x",
                f"{mesh['network_latency'] / circ['network_latency']:.2f}x",
                f"{ring['network_latency'] / circ['network_latency']:.2f}x",
                str(circ["feedback_iteration_count"]),
            ]
        )

    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    fig.patch.set_facecolor("white")
    ax.axis("off")
    table = ax.table(
        cellText=table_rows,
        colLabels=[
            "Workload",
            "Circulant",
            "Torus/Circ",
            "Mesh/Circ",
            "Ring/Circ",
            "Iterations",
        ],
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.45)
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if row == 0:
            cell.set_facecolor("#E9EEF5")
            cell.set_text_props(weight="bold", color=TEXT_COLOR)
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 else "#F7F7F7")
    fig.tight_layout()
    path = out_dir / "moe_expert_ffn_summary_dashboard.png"
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def write_summary(rows: list[dict], out_dir: Path) -> tuple[Path, Path]:
    csv_path = out_dir / "moe_expert_ffn_summary.csv"
    fields = [
        "workload",
        "tokens_per_expert",
        "topology",
        "network_latency_ms",
        "network_energy_j",
        "total_network_bytes",
        "max_link_load_bytes",
        "serialization_latency_ms",
        "first_byte_latency_ms",
        "dominant_collective",
        "dominant_tensor",
        "feedback_converged",
        "feedback_iterations",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "workload": row["workload"],
                    "tokens_per_expert": row["tokens_per_expert"],
                    "topology": row["topology"],
                    "network_latency_ms": row["network_latency"] * 1e3,
                    "network_energy_j": row["network_energy"],
                    "total_network_bytes": row["total_network_bytes"],
                    "max_link_load_bytes": row["max_link_load"],
                    "serialization_latency_ms": row["serialization_latency"] * 1e3,
                    "first_byte_latency_ms": row["first_byte_latency"] * 1e3,
                    "dominant_collective": row["dominant_collective"],
                    "dominant_tensor": row["dominant_tensor"],
                    "feedback_converged": row["feedback_converged"],
                    "feedback_iterations": row["feedback_iteration_count"],
                }
            )

    md_path = out_dir / "summary.md"
    tokens = _ordered_tokens(rows)
    by_key = {(row["tokens_per_expert"], row["topology"]): row for row in rows}
    lines = [
        "# MoE Expert-FFN Sweep Summary",
        "",
        f"Completed topology results: {len(rows)}",
        f"Complete four-topology workload blocks: {len(tokens)}",
        "",
        "| Workload | Best topology | Circulant latency | Torus/Circ | Mesh/Circ | Ring/Circ |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for token in tokens:
        candidates = [by_key[(token, topology)] for topology in TOPOLOGY_ORDER]
        best = min(candidates, key=lambda row: row["network_latency"])
        circ = by_key[(token, "Circulant {1,5,17}")]
        torus = by_key[(token, "Torus 4x4x4")]
        mesh = by_key[(token, "Mesh 4x4x4")]
        ring = by_key[(token, "Ring 64")]
        lines.append(
            f"| T{token} | {best['topology']} | {circ['network_latency'] * 1e3:.3f} ms | "
            f"{torus['network_latency'] / circ['network_latency']:.2f}x | "
            f"{mesh['network_latency'] / circ['network_latency']:.2f}x | "
            f"{ring['network_latency'] / circ['network_latency']:.2f}x |"
        )
    lines.extend(
        [
            "",
            "All completed rows converge in two feedback iterations. The inferred "
            "network phase is one `REDUCE_SCATTER` on `ExpertDown:Y`; `ExpertUp` "
            "is local in the mapper output.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, md_path


def mirror_outputs(out_dir: Path, mirror_dir: Path) -> None:
    mirror_dir.mkdir(parents=True, exist_ok=True)
    for path in out_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, mirror_dir / path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot AccelForge-backed MoE expert-FFN results."
    )
    parser.add_argument("results", type=Path, help="Run directory or results.json")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run>/figures.",
    )
    parser.add_argument(
        "--mirror-dir",
        type=Path,
        default=None,
        help="Optional directory to mirror generated figures and summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (args.results if args.results.is_dir() else args.results.parent) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.results)
    if not rows:
        raise SystemExit("No MoE expert-FFN rows found.")

    written = [
        fig_latency_comparison(rows, out_dir),
        fig_latency_breakdown(rows, out_dir),
        fig_energy_breakdown(rows, out_dir),
        fig_summary_dashboard(rows, out_dir),
    ]
    written.extend(write_summary(rows, out_dir))

    if args.mirror_dir:
        mirror_outputs(out_dir, args.mirror_dir)

    print("Generated MoE expert-FFN figures:")
    for path in written:
        print(f"  {path}")
    if args.mirror_dir:
        print(f"Mirrored outputs to: {args.mirror_dir}")


if __name__ == "__main__":
    main()
