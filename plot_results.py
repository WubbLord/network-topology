#!/usr/bin/env python3
"""
Generate polished, pastel-colored graphs for the network topology study.

Usage:
    python plot_results.py <run-dir>
    # e.g.: python plot_results.py logs/slurm-11826655
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from network_topology import compute_network_cost
from network_topology.cost_model import NetworkTransfer, CollectiveType
from network_topology.topology import Torus3D, Mesh3D, Ring, TorusND
from network_topology.tpu_v4 import ICI_LINK_BW_UNIDIR, ICI_ENERGY_PER_BIT_PER_HOP, ICI_PER_HOP_LATENCY

hw = dict(link_bandwidth=ICI_LINK_BW_UNIDIR, energy_per_bit_per_hop=ICI_ENERGY_PER_BIT_PER_HOP,
          per_hop_latency=ICI_PER_HOP_LATENCY)

# ---- Pastel color palette ----
COLORS = {
    "Circulant {1,5,17}": "#AA96DA",   # lilac (proposed - best)
    "Torus 4x4x4":       "#7EB8DA",   # soft blue
    "4D Torus 4x4x2x2":  "#A8D5BA",   # soft green
    "5D Torus 4x2x2x2x2":"#C4B7E0",   # soft purple
    "Mesh 4x4x4":        "#F4A7A0",   # soft coral
    "Torus 8x2x4":       "#A8D5BA",   # soft green
    "Torus 16x2x2":      "#C4B7E0",   # soft purple
    "Ring 64":            "#F9D986",   # soft gold
}
COMPUTE_COLOR = "#D1E8D0"  # mint green
NETWORK_COLOR = "#F4A7A0"  # soft coral
BG_COLOR      = "#FAFAFA"
GRID_COLOR    = "#E8E8E8"
TEXT_COLOR     = "#333333"
SHOW_FIGURE_TITLES = True

TOPO_ORDER = ["Circulant {1,5,17}", "Torus 4x4x4", "4D Torus 4x4x2x2", "Mesh 4x4x4", "Ring 64"]
TOPO_SHORT = {"Circulant {1,5,17}": "Circulant\n{1,5,17}",
              "Torus 4x4x4": "Torus\n4×4×4",
              "4D Torus 4x4x2x2": "4D Torus\n4×4×2×2", "Mesh 4x4x4": "Mesh\n4×4×4",
              "Torus 8x2x4": "Torus\n8×2×4", "Torus 16x2x2": "Torus\n16×2×2",
              "Ring 64": "Ring\n64",
              "5D Torus 4x2x2x2x2": "5D Torus\n4×2⁴"}

def style_ax(ax, title=None, ylabel=None, xlabel=None, title_is_figure=False):
    """Apply clean styling to an axis."""
    ax.set_facecolor(BG_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(colors=TEXT_COLOR, labelsize=10)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
    if title and (SHOW_FIGURE_TITLES or not title_is_figure):
        ax.set_title(title, fontsize=13, fontweight="bold", color=TEXT_COLOR, pad=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, color=TEXT_COLOR)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, color=TEXT_COLOR)


BASE_WORKLOAD_ORDER = [
    "GPT3 175B",
    "Small 2Kx8K", "Small 4Kx16K", "Medium 8Kx32K", "Medium 16Kx32K",
    "Wide 8Kx256K", "VeryWide 2Kx256K", "Rect 64Kx128K",
    "Square 128Kx128K", "Tall 256Kx64K", "LargeSquare 256Kx256K", "VeryTall 256Kx8K",
]
BASE_WORKLOAD_LABELS = {
    "GPT3 175B": "GPT3\n175B",
    "Small 2Kx8K": "Small\n2K×8K",
    "Small 4Kx16K": "Small\n4K×16K",
    "Medium 8Kx32K": "Medium\n8K×32K",
    "Medium 16Kx32K": "Medium\n16K×32K",
    "Square 128Kx128K": "Square\n128K×128K",
    "Wide 8Kx256K": "Wide\n8K×256K",
    "Tall 256Kx64K": "Tall\n256K×64K",
    "VeryWide 2Kx256K": "VeryWide\n2K×256K",
    "Rect 64Kx128K": "Rect\n64K×128K",
    "LargeSquare 256Kx256K": "LargeSq\n256K×256K",
    "VeryTall 256Kx8K": "VeryTall\n256K×8K",
}
BASE_WORKLOAD_SHORTS = {
    "GPT3 175B": "GPT3",
    "Small 2Kx8K": "2K×8K",
    "Small 4Kx16K": "4K×16K",
    "Medium 8Kx32K": "8K×32K",
    "Medium 16Kx32K": "16K×32K",
    "Square 128Kx128K": "128K×128K",
    "Wide 8Kx256K": "8K×256K",
    "Tall 256Kx64K": "256K×64K",
    "VeryWide 2Kx256K": "2K×256K",
    "Rect 64Kx128K": "64K×128K",
    "LargeSquare 256Kx256K": "256K×256K",
    "VeryTall 256Kx8K": "256K×8K",
}


def _batched_workload_parts(workload_name):
    match = re.match(r"^Batched B(\d+) (.+)$", workload_name)
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def _tall_workload_parts(workload_name):
    match = re.match(r"^(?:B(\d+) )?([0-9]+K)x([0-9]+K)$", workload_name)
    if not match:
        return None
    batch_size, n_size, k_size = match.groups()
    return int(batch_size or 1), n_size, k_size


def _workload_order_key(workload_name):
    batched = _batched_workload_parts(workload_name)
    if batched:
        batch_size, base_name = batched
        try:
            base_idx = BASE_WORKLOAD_ORDER.index(base_name)
        except ValueError:
            base_idx = len(BASE_WORKLOAD_ORDER)
        return (1, batch_size, base_idx, base_name)
    tall = _tall_workload_parts(workload_name)
    if tall:
        batch_size, n_size, k_size = tall
        return (2, batch_size, int(n_size[:-1]), int(k_size[:-1]))
    try:
        return (0, 0, BASE_WORKLOAD_ORDER.index(workload_name), workload_name)
    except ValueError:
        return (3, 0, len(BASE_WORKLOAD_ORDER), workload_name)


def _workload_label(workload_name):
    batched = _batched_workload_parts(workload_name)
    if batched:
        batch_size, base_name = batched
        base_label = BASE_WORKLOAD_LABELS.get(base_name, base_name).replace("\n", " ")
        return f"B{batch_size}\n{base_label}"
    tall = _tall_workload_parts(workload_name)
    if tall:
        batch_size, n_size, k_size = tall
        prefix = "" if batch_size == 1 else f"B{batch_size}\n"
        return f"{prefix}{n_size}x{k_size}"
    return BASE_WORKLOAD_LABELS.get(workload_name, workload_name)


def _workload_short(workload_name):
    batched = _batched_workload_parts(workload_name)
    if batched:
        batch_size, base_name = batched
        base_short = BASE_WORKLOAD_SHORTS.get(base_name, base_name.split()[0])
        return f"B{batch_size}\n{base_short}"
    tall = _tall_workload_parts(workload_name)
    if tall:
        batch_size, n_size, k_size = tall
        prefix = "" if batch_size == 1 else f"B{batch_size}\n"
        return f"{prefix}{n_size}x{k_size}"
    return BASE_WORKLOAD_SHORTS.get(workload_name, workload_name.split()[0])


def _format_dim_value(value, dim_name=None):
    try:
        value = int(value)
    except (TypeError, ValueError):
        return str(value)
    if dim_name == "B":
        return f"{value:,}"
    if value >= 1024 and value % 1024 == 0:
        return f"{value // 1024}K"
    return f"{value:,}"


def _entries_by_workload(entries):
    by_workload = {}
    for (desc, _), (result, _) in entries.items():
        by_workload.setdefault(desc, result.get("params", {}))
    return by_workload


def load_entries(run_dir: Path):
    """Load all per-job JSON results."""
    entries = {}
    skip_files = {"results.json", "milestone2_analysis.json"}

    def add_results(data):
        for r in data.get("results", []):
            for tn, td in r.get("topologies", {}).items():
                entries[(r["desc"], tn)] = (r, td)

    for jf in sorted(run_dir.glob("*.json")):
        if jf.name in skip_files:
            continue
        with jf.open() as f:
            add_results(json.load(f))

    if entries:
        return entries

    results_path = run_dir / "results.json"
    if results_path.exists():
        with results_path.open() as f:
            add_results(json.load(f))

    return entries


# ===========================================================================
# Figure 1: Workload descriptions visual
# ===========================================================================
def fig_workloads(out_dir, entries=None):
    palette = [
        "#D8C7A3", "#B6D7A8", "#A4C2F4", "#D5A6BD", "#F4A7A0",
        "#F9D986", "#C4B7E0", "#7EB8DA", "#A8D5BA", "#6BC5B0", "#E8B4B8",
    ]

    if entries:
        workloads = []
        by_workload = _entries_by_workload(entries)
        for idx, desc in enumerate(sorted(by_workload, key=_workload_order_key)):
            params = by_workload[desc]
            dims = []
            if "BATCH_SIZE" in params and _batched_workload_parts(desc):
                dims.append(("B\n(batch)", params["BATCH_SIZE"], "B"))
            if "M" in params:
                dims.append(("M\n(rows)", params["M"], "M"))
            if "KN" in params:
                dims.append(("K=N\n(cols)", params["KN"], "KN"))
            if "N_TOKENS" in params and not dims:
                dims.append(("Tokens", params["N_TOKENS"], "N_TOKENS"))
            if "N_LAYERS" in params and not dims:
                dims.append(("Layers", params["N_LAYERS"], "N_LAYERS"))
            if dims:
                workloads.append((_workload_label(desc), dims, palette[idx % len(palette)]))
    else:
        workloads = [
            ("Small\n2K×8K", [("M\n(rows)", 2048, "M"), ("K=N\n(cols)", 8192, "KN")], "#D8C7A3"),
            ("Small\n4K×16K", [("M\n(rows)", 4096, "M"), ("K=N\n(cols)", 16384, "KN")], "#B6D7A8"),
            ("Medium\n8K×32K", [("M\n(rows)", 8192, "M"), ("K=N\n(cols)", 32768, "KN")], "#A4C2F4"),
            ("Medium\n16K×32K", [("M\n(rows)", 16384, "M"), ("K=N\n(cols)", 32768, "KN")], "#D5A6BD"),
            ("Wide (FFN)\n8K×256K", [("M\n(rows)", 8192, "M"), ("K=N\n(cols)", 262144, "KN")], "#F4A7A0"),
            ("VeryWide\n2K×256K", [("M\n(rows)", 2048, "M"), ("K=N\n(cols)", 262144, "KN")], "#F9D986"),
            ("Rect\n64K×128K", [("M\n(rows)", 65536, "M"), ("K=N\n(cols)", 131072, "KN")], "#C4B7E0"),
            ("Square\n128K×128K", [("M\n(rows)", 131072, "M"), ("K=N\n(cols)", 131072, "KN")], "#7EB8DA"),
            ("Tall\n256K×64K", [("M\n(rows)", 262144, "M"), ("K=N\n(cols)", 65536, "KN")], "#A8D5BA"),
            ("LargeSq\n256K×256K", [("M\n(rows)", 262144, "M"), ("K=N\n(cols)", 262144, "KN")], "#6BC5B0"),
            ("VeryTall\n256K×8K", [("M\n(rows)", 262144, "M"), ("K=N\n(cols)", 8192, "KN")], "#E8B4B8"),
        ]
    if not workloads:
        return

    ncols = 4
    nrows = (len(workloads) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 3.5 * nrows))
    fig.patch.set_facecolor("white")
    if SHOW_FIGURE_TITLES:
        fig.suptitle("Workloads Tested", fontsize=16, fontweight="bold", color=TEXT_COLOR, y=0.98)

    flat_axes = axes.flatten()
    for i, (label, dims, color) in enumerate(workloads):
        ax = flat_axes[i]
        ax.set_facecolor(BG_COLOR)
        dim_labels = [d[0] for d in dims]
        dim_values = [int(d[1]) for d in dims]
        dim_names = [d[2] for d in dims]
        bars = ax.bar(dim_labels, dim_values, color=[color] * len(dim_values),
                      edgecolor="white", linewidth=2, width=0.5, zorder=3, alpha=0.85)
        ax.set_title(label, fontsize=11, fontweight="bold", color=TEXT_COLOR, pad=8)
        if i % ncols == 0:
            ax.set_ylabel("Dimension size", fontsize=9, color=TEXT_COLOR)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: _format_dim_value(x)))
        style_ax(ax)
        max_dim = max(dim_values)
        for bar, val, dim_name in zip(bars, dim_values, dim_names):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max_dim * 0.03,
                    _format_dim_value(val, dim_name), ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color=TEXT_COLOR)

    # Hide unused subplot
    for j in range(len(workloads), nrows * ncols):
        flat_axes[j].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.92 if SHOW_FIGURE_TITLES else 1])
    path = out_dir / "1_workloads.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


# ===========================================================================
# Figure 2: Topology properties comparison
# ===========================================================================
def fig_topologies(out_dir):
    from network_topology.topology import CirculantHD
    topos = {
        "Circulant {1,5,17}": CirculantHD(64, (1,5,17), **hw),
        "Torus 4x4x4": Torus3D(dims=(4,4,4), **hw),
        "4D Torus 4x4x2x2": TorusND(dims=(4,4,2,2), **hw),
        "Mesh 4x4x4": Mesh3D(dims=(4,4,4), **hw),
        "Torus 8x2x4": Torus3D(dims=(8,2,4), **hw),
        "Torus 16x2x2": Torus3D(dims=(16,2,2), **hw),
        "Ring 64": Ring(num_chips=64, **hw),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor("white")
    if SHOW_FIGURE_TITLES:
        fig.suptitle("Network Topology Properties (64 chips)", fontsize=16,
                     fontweight="bold", color=TEXT_COLOR, y=0.98)

    names = list(TOPO_ORDER)
    short_names = [TOPO_SHORT[n] for n in names]
    colors = [COLORS[n] for n in names]

    # Diameter
    vals = [topos[n].summary()["diameter"] for n in names]
    bars = axes[0].bar(short_names, vals, color=colors, edgecolor="white", linewidth=2, zorder=3)
    style_ax(axes[0], "Diameter (max hops)", "Hops")
    for bar, v in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width()/2, v + 0.5, str(v),
                     ha="center", fontsize=10, fontweight="bold", color=TEXT_COLOR)

    # Degree
    vals = [topos[n].summary()["min_degree"] for n in names]
    bars = axes[1].bar(short_names, vals, color=colors, edgecolor="white", linewidth=2, zorder=3)
    style_ax(axes[1], "Links per Chip (degree)", "Links")
    for bar, v in zip(bars, vals):
        axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.1, str(v),
                     ha="center", fontsize=10, fontweight="bold", color=TEXT_COLOR)

    # Bisection BW
    vals = [topos[n].summary()["bisection_bandwidth_TB_s"] for n in names]
    bars = axes[2].bar(short_names, vals, color=colors, edgecolor="white", linewidth=2, zorder=3)
    style_ax(axes[2], "Bisection Bandwidth", "TB/s")
    for bar, v in zip(bars, vals):
        axes[2].text(bar.get_x() + bar.get_width()/2, v + 0.03, f"{v:.2f}",
                     ha="center", fontsize=10, fontweight="bold", color=TEXT_COLOR)

    plt.tight_layout(rect=[0, 0, 1, 0.92 if SHOW_FIGURE_TITLES else 1])
    path = out_dir / "2_topologies.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


# ===========================================================================
# Figure 3: Network latency comparison (grouped bars)
# ===========================================================================
def _available_workloads(entries):
    """Find all workloads with at least 2 topology results, in a stable order."""
    wl_with_data = []
    for wl in sorted({desc for desc, _ in entries}, key=_workload_order_key):
        count = sum(1 for tn in TOPO_ORDER if (wl, tn) in entries)
        if count >= 2:
            wl_with_data.append(wl)
    labels = [_workload_label(w) for w in wl_with_data]
    return wl_with_data, labels


def fig_latency_comparison(entries, out_dir):
    workloads, wl_labels = _available_workloads(entries)
    if not workloads:
        return

    fig, ax = plt.subplots(figsize=(max(12, 2.5 * len(workloads)), 6))
    fig.patch.set_facecolor("white")

    avail_topos = [tn for tn in TOPO_ORDER if any((wl, tn) in entries for wl in workloads)]
    x = np.arange(len(workloads))
    n_topos = len(avail_topos)
    width = min(0.15, 0.8 / n_topos)
    offsets = np.arange(n_topos) - (n_topos - 1) / 2

    for i, tn in enumerate(avail_topos):
        vals = []
        for wl in workloads:
            if (wl, tn) in entries:
                vals.append(entries[(wl, tn)][1]["total_latency"] * 1e3)
            else:
                vals.append(0)
        ax.bar(x + offsets[i] * width, vals, width * 0.9, label=tn,
               color=COLORS.get(tn, "#CCCCCC"), edgecolor="white", linewidth=1.5, zorder=3)

    style_ax(
        ax,
        "Network Latency by Workload and Topology",
        "Network Latency (ms)",
        title_is_figure=True,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(wl_labels, fontsize=10)
    ax.legend(frameon=True, facecolor="white", edgecolor="#DDDDDD", fontsize=10,
              loc="upper left", ncol=min(n_topos, 4))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    plt.tight_layout()
    path = out_dir / "3_latency_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


# ===========================================================================
# Figure 4: Energy breakdown (compute vs network, stacked)
# ===========================================================================
def fig_energy_breakdown(entries, out_dir):
    workloads, wl_labels = _available_workloads(entries)
    if not workloads:
        return

    ncols = min(4, len(workloads))
    nrows = (len(workloads) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5.5 * nrows), sharey=False)
    fig.patch.set_facecolor("white")
    if SHOW_FIGURE_TITLES:
        fig.suptitle("Energy Breakdown: Compute vs Network", fontsize=16,
                     fontweight="bold", color=TEXT_COLOR, y=0.98)

    flat_axes = np.atleast_1d(axes).flatten() if nrows > 1 or ncols > 1 else [axes]

    for idx, (wl, wl_label) in enumerate(zip(workloads, wl_labels)):
        ax = flat_axes[idx]
        names = []
        compute_vals = []
        network_vals = []

        for tn in TOPO_ORDER:
            if (wl, tn) not in entries:
                continue
            td = entries[(wl, tn)][1]
            names.append(TOPO_SHORT[tn])
            compute_vals.append(td["compute_energy"])
            network_vals.append(td["total_energy"])

        x = np.arange(len(names))
        ax.bar(x, compute_vals, 0.55, label="Compute" if idx == 0 else "",
               color=COMPUTE_COLOR, edgecolor="white", linewidth=1.5, zorder=3)
        ax.bar(x, network_vals, 0.55, bottom=compute_vals,
               label="Network" if idx == 0 else "",
               color=NETWORK_COLOR, edgecolor="white", linewidth=1.5, zorder=3, alpha=0.85)

        for i in range(len(names)):
            total = compute_vals[i] + network_vals[i]
            net_pct = network_vals[i] / total * 100
            ax.text(i, total + total * 0.02, f"{net_pct:.0f}%",
                    ha="center", fontsize=9, color=TEXT_COLOR, fontweight="bold")

        style_ax(ax, wl_label.replace("\n", " "), "Energy (J)" if idx % ncols == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8)

    flat_axes[0].legend(frameon=True, facecolor="white", edgecolor="#DDDDDD",
                        fontsize=10, loc="upper left")

    for j in range(len(workloads), len(flat_axes)):
        flat_axes[j].set_visible(False)

    fig.text(0.5, 0.01, "Percentages show network share of total energy",
             ha="center", fontsize=10, color="#888888", style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 0.92 if SHOW_FIGURE_TITLES else 1])
    path = out_dir / "4_energy_breakdown.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


# ===========================================================================
# Figure 5: Latency breakdown (compute vs network, horizontal stacked)
# ===========================================================================
def fig_latency_breakdown(entries, out_dir):
    workloads, _ = _available_workloads(entries)

    labels = []
    compute_vals = []
    network_vals = []

    for wl in workloads:
        for tn in reversed(TOPO_ORDER):
            if (wl, tn) not in entries:
                continue
            td = entries[(wl, tn)][1]
            short_wl = _workload_short(wl)
            labels.append(f"{short_wl} — {tn}")
            compute_vals.append(td["compute_latency"])
            network_vals.append(td["total_latency"])

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(14, max(6, 0.5 * len(labels))))
    fig.patch.set_facecolor("white")

    y = np.arange(len(labels))
    ax.barh(y, compute_vals, 0.6, label="Compute",
            color=COMPUTE_COLOR, edgecolor="white", linewidth=1.5, zorder=3)
    ax.barh(y, network_vals, 0.6, left=compute_vals, label="Network",
            color=NETWORK_COLOR, edgecolor="white", linewidth=1.5, zorder=3, alpha=0.85)

    for i in range(len(labels)):
        total = compute_vals[i] + network_vals[i]
        net_pct = network_vals[i] / total * 100
        ax.text(total + total * 0.01, i, f" {net_pct:.0f}% net",
                va="center", fontsize=8, color="#888888")

    style_ax(
        ax,
        "Latency Breakdown: Compute vs Network",
        "Total Latency (s)",
        title_is_figure=True,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Latency (seconds)", fontsize=11, color=TEXT_COLOR)
    ax.legend(frameon=True, facecolor="white", edgecolor="#DDDDDD", fontsize=10, loc="lower right")
    ax.invert_yaxis()

    plt.tight_layout()
    path = out_dir / "5_latency_breakdown.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


# ===========================================================================
# Figure 6: Torus aspect ratio comparison
# ===========================================================================
def fig_torus_aspect(entries, out_dir):
    torus_topos = ["Torus 4x4x4", "Torus 8x2x4", "Torus 16x2x2", "Ring 64"]
    torus_labels = ["4×4×4\n(cube)", "8×2×4", "16×2×2", "Ring 64"]
    torus_colors = [COLORS[t] for t in torus_topos]

    # Use all workloads that have Torus 4x4x4 as baseline
    workloads, wl_labels = _available_workloads(entries)
    workloads = [wl for wl in workloads if (wl, "Torus 4x4x4") in entries]
    if not workloads:
        return
    wl_short = [_workload_short(wl) for wl in workloads]

    fig, ax = plt.subplots(figsize=(max(10, 0.85 * len(workloads)), 5.5))
    fig.patch.set_facecolor("white")

    x = np.arange(len(workloads))
    width = min(0.18, 0.8 / len(torus_topos))

    for i, (tn, color) in enumerate(zip(torus_topos, torus_colors)):
        vals = []
        for wl in workloads:
            if (wl, tn) in entries:
                base = entries[(wl, "Torus 4x4x4")][1]["total_latency"]
                val = entries[(wl, tn)][1]["total_latency"]
                vals.append(val / base)
            else:
                vals.append(0)
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, vals, width * 0.85, label=torus_labels[i],
                      color=color, edgecolor="white", linewidth=1.5, zorder=3)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                        f"{v:.2f}×", ha="center", fontsize=8, color=TEXT_COLOR, fontweight="bold")

    style_ax(
        ax,
        "Torus Aspect Ratio Impact on Latency",
        "Latency (normalized to 4×4×4 cube)",
        title_is_figure=True,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(wl_short, fontsize=10)
    ax.axhline(y=1.0, color="#999999", linestyle="--", linewidth=1, zorder=2, alpha=0.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#DDDDDD", fontsize=10,
              title="Topology", title_fontsize=11)
    ax.set_ylim(0, max(1.5, ax.get_ylim()[1] * 1.1))

    plt.tight_layout()
    path = out_dir / "6_torus_aspect_ratio.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


# ===========================================================================
# Figure 7: GPT-3 stress analysis (direct model)
# ===========================================================================
def fig_gpt3_stress(out_dir):
    from network_topology.topology import CirculantHD
    topos = {
        "Circulant {1,5,17}": CirculantHD(64, (1,5,17), **hw),
        "Torus 4x4x4": Torus3D(dims=(4,4,4), **hw),
        "Mesh 4x4x4": Mesh3D(dims=(4,4,4), **hw),
        "Ring 64": Ring(num_chips=64, **hw),
    }

    configs = [
        ("GPT3-175B", 96, 12288, 2),
        ("GPT3-6.7B", 32, 4096, 2),
        ("GPT3-1.3B", 24, 2048, 2),
    ]
    scenarios = [
        ("Prefill\n1×2K", 1, 2048),
        ("Prefill\n1×8K", 1, 8192),
        ("Prefill\n1×32K", 1, 32768),
        ("Batch\n32×2K", 32, 2048),
        ("Decode\n256×1", 256, 1),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)
    fig.patch.set_facecolor("white")
    if SHOW_FIGURE_TITLES:
        fig.suptitle("GPT-3 Network Latency by Model Size and Scenario",
                     fontsize=16, fontweight="bold", color=TEXT_COLOR, y=0.98)

    gpt_colors = {"Circulant {1,5,17}": COLORS["Circulant {1,5,17}"],
                  "Torus 4x4x4": COLORS["Torus 4x4x4"],
                  "Mesh 4x4x4": COLORS["Mesh 4x4x4"],
                  "Ring 64": COLORS["Ring 64"]}
    gpt_topos = ["Circulant {1,5,17}", "Torus 4x4x4", "Mesh 4x4x4", "Ring 64"]

    for ax_idx, (model_name, layers, hidden, bpv) in enumerate(configs):
        ax = axes[ax_idx]
        x = np.arange(len(scenarios))
        n_gt = len(gpt_topos)
        width = 0.8 / n_gt

        for ti, tn in enumerate(gpt_topos):
            topo = topos[tn]
            vals = []
            for _, batch, seq in scenarios:
                activation = batch * seq * hidden * bpv
                transfers = [NetworkTransfer(f"L{l}:{op}", activation, CollectiveType.ALLREDUCE)
                             for l in range(layers) for op in ["attn", "ffn"]]
                cost = compute_network_cost(topo, transfers)
                vals.append(cost.total_latency * 1e3)

            offset = (ti - (n_gt - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width * 0.85, label=tn if ax_idx == 0 else "",
                          color=gpt_colors[tn], edgecolor="white", linewidth=1.5, zorder=3)

        style_ax(ax, model_name, "Latency (ms)" if ax_idx == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels([s[0] for s in scenarios], fontsize=9)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    axes[0].legend(frameon=True, facecolor="white", edgecolor="#DDDDDD", fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.92 if SHOW_FIGURE_TITLES else 1])
    path = out_dir / "7_gpt3_stress.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


# ===========================================================================
# Figure 8: Summary dashboard
# ===========================================================================
def fig_summary(entries, out_dir):
    workloads, wl_labels = _available_workloads(entries)
    n_wl = len(workloads)
    if not workloads:
        return

    fig = plt.figure(figsize=(max(16, 0.75 * n_wl), 10))
    fig.patch.set_facecolor("white")
    if SHOW_FIGURE_TITLES:
        fig.suptitle("Network Topology Impact — Summary Dashboard",
                     fontsize=18, fontweight="bold", color=TEXT_COLOR, y=0.97)

    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3,
                          left=0.06, right=0.97,
                          top=0.90 if SHOW_FIGURE_TITLES else 0.96,
                          bottom=0.06)

    # --- Panel 1: Winner summary ---
    ax1 = fig.add_subplot(gs[0, 0])
    topos_tested = TOPO_ORDER
    wins = defaultdict(int)
    for wl in workloads:
        lats = {tn: entries[(wl, tn)][1]["total_latency"]
                for tn in topos_tested if (wl, tn) in entries}
        if lats:
            winner = min(lats, key=lats.get)
            wins[winner] += 1

    win_counts = [wins.get(tn, 0) for tn in topos_tested]
    colors = [COLORS[tn] for tn in topos_tested]
    bars = ax1.bar([TOPO_SHORT[t] for t in topos_tested], win_counts,
                   color=colors, edgecolor="white", linewidth=2, zorder=3)
    style_ax(ax1, f"Workloads Won (of {n_wl})", "Count")
    ax1.set_ylim(0, n_wl + 1)
    for bar, v in zip(bars, win_counts):
        if v > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, v + 0.1, str(v),
                     ha="center", fontsize=12, fontweight="bold", color=TEXT_COLOR)

    # --- Panel 2: Network % of total latency ---
    ax2 = fig.add_subplot(gs[0, 1])
    wl_short = [_workload_short(wl) for wl in workloads]
    for tn in ["Torus 4x4x4", "Ring 64"]:
        pcts = []
        pct_labels = []
        for wl, sl in zip(workloads, wl_short):
            if (wl, tn) in entries:
                td = entries[(wl, tn)][1]
                tl = td["compute_latency"] + td["total_latency"]
                pcts.append(td["total_latency"] / tl * 100)
                pct_labels.append(sl)
        if pcts:
            ax2.plot(pct_labels, pcts, "o-", color=COLORS[tn], linewidth=2.5,
                     markersize=7, label=tn, zorder=3)
    style_ax(ax2, "Network % of Total Latency")
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax2.legend(frameon=True, facecolor="white", edgecolor="#DDDDDD", fontsize=9)
    ax2.tick_params(axis="x", rotation=45)

    # --- Panel 3: Speedup vs Ring ---
    ax3 = fig.add_subplot(gs[0, 2])
    for tn in ["Torus 4x4x4", "Mesh 4x4x4"]:
        speedups = []
        sp_labels = []
        for wl, sl in zip(workloads, wl_short):
            if (wl, tn) in entries and (wl, "Ring 64") in entries:
                ring_lat = entries[(wl, "Ring 64")][1]["total_latency"]
                topo_lat = entries[(wl, tn)][1]["total_latency"]
                speedups.append(ring_lat / topo_lat)
                sp_labels.append(sl)
        if speedups:
            ax3.plot(sp_labels, speedups, "o-", color=COLORS[tn], linewidth=2.5,
                     markersize=7, label=tn, zorder=3)
    style_ax(ax3, "Speedup vs Ring 64")
    ax3.axhline(y=1.0, color="#999999", linestyle="--", linewidth=1, alpha=0.5)
    ax3.legend(frameon=True, facecolor="white", edgecolor="#DDDDDD", fontsize=8)
    ax3.tick_params(axis="x", rotation=45)

    # --- Panel 4: Energy comparison (bottom left) ---
    ax4 = fig.add_subplot(gs[1, 0:2])
    avail_topos = [tn for tn in TOPO_ORDER if any((wl, tn) in entries for wl in workloads)]
    x = np.arange(n_wl)
    width = min(0.15, 0.8 / max(len(avail_topos), 1))
    for i, tn in enumerate(avail_topos):
        vals = []
        for wl in workloads:
            if (wl, tn) in entries:
                td = entries[(wl, tn)][1]
                vals.append(td["compute_energy"] + td["total_energy"])
            else:
                vals.append(0)
        offset = (i - (len(avail_topos) - 1) / 2) * width
        ax4.bar(x + offset, vals, width * 0.85, label=tn, color=COLORS.get(tn, "#CCC"),
                edgecolor="white", linewidth=1.5, zorder=3)
    style_ax(ax4, "Total Energy (Compute + Network)", "Energy (J)")
    ax4.set_xticks(x)
    ax4.set_xticklabels(wl_short, fontsize=10)
    ax4.legend(frameon=True, facecolor="white", edgecolor="#DDDDDD", fontsize=9,
               ncol=min(len(avail_topos), 4))
    ax4.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))

    # --- Panel 5: Key insight text (auto-generated) ---
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")

    # Compute insights dynamically
    best_topo_counts = defaultdict(int)
    for wl in workloads:
        lats = {tn: entries[(wl, tn)][1]["total_latency"]
                for tn in TOPO_ORDER if (wl, tn) in entries}
        if lats:
            best_topo_counts[min(lats, key=lats.get)] += 1
    overall_winner = max(best_topo_counts, key=best_topo_counts.get) if best_topo_counts else "N/A"
    winner_count = best_topo_counts.get(overall_winner, 0)

    # Network % range
    net_pcts = []
    for wl in workloads:
        for tn in TOPO_ORDER:
            if (wl, tn) in entries:
                td = entries[(wl, tn)][1]
                tl = td["compute_latency"] + td["total_latency"]
                if tl > 0:
                    net_pcts.append(td["total_latency"] / tl * 100)
    net_range = f"{min(net_pcts):.0f}–{max(net_pcts):.0f}" if net_pcts else "N/A"

    # Torus vs Ring speedup
    torus_speedups = []
    for wl in workloads:
        if (wl, "Torus 4x4x4") in entries and (wl, "Ring 64") in entries:
            r = entries[(wl, "Ring 64")][1]["total_latency"]
            t = entries[(wl, "Torus 4x4x4")][1]["total_latency"]
            if t > 0:
                torus_speedups.append(r / t)
    avg_torus_speedup = sum(torus_speedups) / len(torus_speedups) if torus_speedups else 0

    insight_text = (
        f"Key Findings\n"
        f"━━━━━━━━━━━━━━━━\n\n"
        f"▸ {overall_winner} wins\n"
        f"  {winner_count}/{n_wl} workloads\n\n"
        f"▸ Network is {net_range}%\n"
        f"  of total latency\n\n"
        f"▸ Torus vs Ring:\n"
        f"  {avg_torus_speedup:.2f}× faster avg\n\n"
        f"▸ {n_wl} workloads tested"
    )
    ax5.text(0.1, 0.95, insight_text, transform=ax5.transAxes,
             fontsize=11, verticalalignment="top", fontfamily="monospace",
             color=TEXT_COLOR,
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#F0F4FF",
                       edgecolor="#C8D8F0", alpha=0.9))

    path = out_dir / "8_summary_dashboard.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <run-dir>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    entries = load_entries(run_dir)
    if not entries:
        print(f"No results found in {run_dir}")
        sys.exit(1)

    out_dir = run_dir / "figures"
    out_dir.mkdir(exist_ok=True)

    print(f"Generating figures in {out_dir}/...")
    fig_workloads(out_dir, entries)
    fig_topologies(out_dir)
    fig_latency_comparison(entries, out_dir)
    fig_energy_breakdown(entries, out_dir)
    fig_latency_breakdown(entries, out_dir)
    fig_torus_aspect(entries, out_dir)
    fig_gpt3_stress(out_dir)
    fig_summary(entries, out_dir)

    print(f"\nDone! {len(list(out_dir.glob('*.png')))} figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
