#!/usr/bin/env python3
"""
Generate aesthetic spatial visualizations of network topologies and matrix workloads.

Produces:
  - 3D Torus, 3D Mesh, Ring topology diagrams with node/edge rendering
  - Matrix workload shape diagrams (Square, Wide, Tall)
  - Combined overview figure

Usage:
    python plot_topologies.py [output-dir]
    # default output: figures/
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np
import networkx as nx

# ---- Pastel palette ----
PAL = {
    "blue":    "#7EB8DA",
    "coral":   "#F4A7A0",
    "green":   "#A8D5BA",
    "purple":  "#C4B7E0",
    "gold":    "#F9D986",
    "pink":    "#FCBAD3",
    "mint":    "#B5EAD7",
    "peach":   "#FFD3B6",
    "sky":     "#A8D8EA",
    "lilac":   "#AA96DA",
}
BG = "#FAFAFA"
TEXT = "#333333"
EDGE_NORMAL = "#B0B0B0"
EDGE_WRAP = "#E8A0A0"


def style_fig(fig):
    fig.patch.set_facecolor("white")


# ===========================================================================
# 3D Grid helpers
# ===========================================================================
def grid_nodes_3d(dims):
    """Generate node positions on a 3D integer grid, centered at origin."""
    dx, dy, dz = dims
    nodes = {}
    idx = 0
    for x in range(dx):
        for y in range(dy):
            for z in range(dz):
                nodes[idx] = np.array([
                    x - (dx - 1) / 2,
                    y - (dy - 1) / 2,
                    z - (dz - 1) / 2
                ])
                idx += 1
    return nodes


def mesh_edges(dims):
    """Edges for a 3D mesh (no wraparound)."""
    dx, dy, dz = dims
    edges = []
    for x in range(dx):
        for y in range(dy):
            for z in range(dz):
                idx = x * dy * dz + y * dz + z
                if x + 1 < dx:
                    edges.append((idx, (x+1)*dy*dz + y*dz + z, False))
                if y + 1 < dy:
                    edges.append((idx, x*dy*dz + (y+1)*dz + z, False))
                if z + 1 < dz:
                    edges.append((idx, x*dy*dz + y*dz + (z+1), False))
    return edges


def torus_extra_edges(dims):
    """Wraparound edges for a 3D torus."""
    dx, dy, dz = dims
    edges = []
    for x in range(dx):
        for y in range(dy):
            for z in range(dz):
                idx = x * dy * dz + y * dz + z
                # Wraparound in each dimension
                if x == dx - 1 and dx > 1:
                    edges.append((idx, 0*dy*dz + y*dz + z, True))
                if y == dy - 1 and dy > 1:
                    edges.append((idx, x*dy*dz + 0*dz + z, True))
                if z == dz - 1 and dz > 1:
                    edges.append((idx, x*dy*dz + y*dz + 0, True))
    return edges


def draw_3d_topology(ax, nodes, edges, title, node_color, wrap_color=EDGE_WRAP,
                     elev=22, azim=135, node_size=120):
    """Draw a 3D topology on an Axes3D."""
    # Draw edges
    for src, dst, is_wrap in edges:
        p0 = nodes[src]
        p1 = nodes[dst]
        style = {"color": wrap_color if is_wrap else EDGE_NORMAL,
                 "linewidth": 1.8 if is_wrap else 1.2,
                 "linestyle": "--" if is_wrap else "-",
                 "alpha": 0.7 if is_wrap else 0.5,
                 "zorder": 1}

        if is_wrap:
            # Draw wraparound as a curved arc through a midpoint pulled outward
            mid = (p0 + p1) / 2
            # Push midpoint outward from center
            center = np.zeros(3)
            direction = mid - center
            norm = np.linalg.norm(direction)
            if norm > 0.1:
                offset = direction / norm * 0.6
            else:
                offset = np.array([0, 0, 0.6])
            ctrl = mid + offset

            t = np.linspace(0, 1, 20)
            curve = np.outer((1-t)**2, p0) + np.outer(2*(1-t)*t, ctrl) + np.outer(t**2, p1)
            ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], **style)
        else:
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], **style)

    # Draw nodes
    coords = np.array([nodes[i] for i in sorted(nodes)])
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               s=node_size, c=node_color, edgecolors="white",
               linewidths=1.5, alpha=0.95, zorder=5, depthshade=True)

    ax.set_title(title, fontsize=14, fontweight="bold", color=TEXT, pad=15)
    ax.view_init(elev=elev, azim=azim)

    # Clean up axes
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#E0E0E0")
    ax.yaxis.pane.set_edgecolor("#E0E0E0")
    ax.zaxis.pane.set_edgecolor("#E0E0E0")
    ax.grid(True, alpha=0.15)


# ===========================================================================
# Figure 1: All topologies side by side
# ===========================================================================
def fig_all_topologies(out_dir):
    fig = plt.figure(figsize=(20, 6))
    style_fig(fig)
    fig.suptitle("Network Topologies for 64-Chip Distributed System",
                 fontsize=18, fontweight="bold", color=TEXT, y=0.98)

    # --- Ring 64 (2D circular) ---
    ax1 = fig.add_subplot(1, 4, 1)
    G = nx.cycle_graph(64)
    pos = nx.circular_layout(G)
    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color=EDGE_NORMAL, width=1.0, alpha=0.4)
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=50, node_color=PAL["gold"],
                           edgecolors="white", linewidths=1.0, alpha=0.9)
    # Highlight a few nodes
    highlight = [0, 16, 32, 48]
    nx.draw_networkx_nodes(G, pos, nodelist=highlight, ax=ax1, node_size=90,
                           node_color=PAL["gold"], edgecolors=PAL["coral"], linewidths=2.0)
    nx.draw_networkx_labels(G, pos, {n: str(n) for n in highlight}, ax=ax1,
                            font_size=7, font_color=TEXT, font_weight="bold")
    ax1.set_title("Ring 64\n2 links/chip, diameter=32", fontsize=12,
                  fontweight="bold", color=TEXT)
    ax1.axis("off")

    # --- Mesh 4x4x4 (3D) ---
    ax2 = fig.add_subplot(1, 4, 2, projection="3d")
    dims = (4, 4, 4)
    nodes = grid_nodes_3d(dims)
    edges = mesh_edges(dims)
    draw_3d_topology(ax2, nodes, edges, "Mesh 4×4×4\n3-6 links/chip, diameter=9",
                     PAL["coral"])

    # --- Torus 4x4x4 (3D with wraparound) ---
    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    mesh_e = mesh_edges(dims)
    wrap_e = torus_extra_edges(dims)
    all_edges = mesh_e + wrap_e
    draw_3d_topology(ax3, nodes, all_edges, "Torus 4×4×4\n6 links/chip, diameter=6",
                     PAL["blue"], wrap_color=PAL["coral"])

    # --- Torus 8x2x4 (3D elongated) ---
    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    dims2 = (4, 2, 8)  # Swap for better visual
    nodes2 = grid_nodes_3d(dims2)
    mesh_e2 = mesh_edges(dims2)
    wrap_e2 = torus_extra_edges(dims2)
    all_e2 = mesh_e2 + wrap_e2
    draw_3d_topology(ax4, nodes2, all_e2, "Torus 8×2×4\n5 links/chip, diameter=7",
                     PAL["green"], wrap_color=PAL["coral"], elev=20, azim=125)

    # Legend
    normal_patch = mpatches.Patch(color=EDGE_NORMAL, label="Direct links")
    wrap_patch = mpatches.Patch(color=PAL["coral"], label="Wraparound links (torus)")
    fig.legend(handles=[normal_patch, wrap_patch], loc="lower center", ncol=2,
               fontsize=11, frameon=True, facecolor="white", edgecolor="#DDDDDD")

    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    path = out_dir / "topologies_spatial.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


# ===========================================================================
# Figure 2: Detailed torus visualization (small 3x3x3 for clarity)
# ===========================================================================
def fig_torus_detail(out_dir):
    fig = plt.figure(figsize=(14, 6))
    style_fig(fig)
    fig.suptitle("Mesh vs Torus: The Impact of Wraparound Links",
                 fontsize=16, fontweight="bold", color=TEXT, y=0.98)

    dims = (3, 3, 3)  # Small for clarity
    nodes = grid_nodes_3d(dims)

    # Mesh
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    edges = mesh_edges(dims)
    draw_3d_topology(ax1, nodes, edges,
                     "3D Mesh (3×3×3)\nNo wraparound — edge nodes have fewer links",
                     PAL["coral"], node_size=200, elev=25, azim=130)

    # Torus
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    wrap = torus_extra_edges(dims)
    all_e = edges + wrap
    draw_3d_topology(ax2, nodes, all_e,
                     "3D Torus (3×3×3)\nWraparound links — every node has 6 links",
                     PAL["blue"], wrap_color=PAL["coral"], node_size=200, elev=25, azim=130)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    path = out_dir / "mesh_vs_torus_detail.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


# ===========================================================================
# Figure 3: Matrix workload shapes (3D box representations)
# ===========================================================================
def draw_box_3d(ax, origin, size, color, alpha=0.15, edge_alpha=0.6, label=None):
    """Draw a 3D rectangular prism."""
    ox, oy, oz = origin
    dx, dy, dz = size

    # Define the 6 faces
    vertices = np.array([
        [ox, oy, oz], [ox+dx, oy, oz], [ox+dx, oy+dy, oz], [ox, oy+dy, oz],
        [ox, oy, oz+dz], [ox+dx, oy, oz+dz], [ox+dx, oy+dy, oz+dz], [ox, oy+dy, oz+dz]
    ])
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
    ]
    poly = Poly3DCollection(faces, alpha=alpha, facecolor=color,
                            edgecolor=color, linewidth=1.5)
    ax.add_collection3d(poly)

    # Draw edges more prominently
    edges = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
    ]
    for e in edges:
        p = vertices[e]
        ax.plot3D(p[:,0], p[:,1], p[:,2], color=color, alpha=edge_alpha, linewidth=1.8)


def fig_matrix_workloads(out_dir):
    fig = plt.figure(figsize=(18, 6))
    style_fig(fig)
    fig.suptitle("Matrix Workload Shapes: T₁ = T₀ × W₀",
                 fontsize=16, fontweight="bold", color=TEXT, y=0.98)

    workloads = [
        ("Square (128K × 128K)", 1.0, 1.0, 1.0),
        ("Wide / FFN-like (8K × 256K)", 0.25, 1.0, 1.0),
        ("Tall (256K × 64K)", 1.0, 0.5, 0.5),
    ]

    for idx, (title, m_scale, k_scale, n_scale) in enumerate(workloads):
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")

        M = m_scale * 2.0
        K = k_scale * 2.0
        N = n_scale * 2.0

        gap = 0.4

        # T0: M x K (input)
        draw_box_3d(ax, (0, 0, 0), (0.2, K, M), PAL["blue"], alpha=0.2, label="T₀")
        # W0: K x N (weight)
        draw_box_3d(ax, (0.2 + gap, 0, 0), (N, 0.2, K), PAL["coral"], alpha=0.2, label="W₀")
        # T1: M x N (output)
        draw_box_3d(ax, (0.2 + gap + N + gap, 0, 0), (0.2, N, M), PAL["green"], alpha=0.2, label="T₁")

        # Labels
        ax.text(0.1, K/2, M + 0.2, "T₀\n(input)", ha="center", fontsize=10,
                fontweight="bold", color=PAL["blue"])
        ax.text(0.2 + gap + N/2, 0.1, K + 0.2, "W₀\n(weight)", ha="center", fontsize=10,
                fontweight="bold", color=PAL["coral"])
        ax.text(0.2 + gap + N + gap + 0.1, N/2, M + 0.2, "T₁\n(output)", ha="center",
                fontsize=10, fontweight="bold", color="#5A9E6F")

        # Dimension annotations
        total_width = 0.2 + gap + N + gap + 0.2
        ax.text(0.1, -0.4, M/2, f"M={int(m_scale*128)}K", ha="center", fontsize=9,
                color=TEXT, style="italic")
        ax.text(0.1, K/2, -0.3, f"K={int(k_scale*128)}K", ha="center", fontsize=9,
                color=TEXT, style="italic")

        ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT, pad=10)
        ax.view_init(elev=20, azim=135)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("#E0E0E0")
        ax.yaxis.pane.set_edgecolor("#E0E0E0")
        ax.zaxis.pane.set_edgecolor("#E0E0E0")
        ax.grid(True, alpha=0.1)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    path = out_dir / "matrix_workload_shapes.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


# ===========================================================================
# Figure 4: Ring topology with data flow
# ===========================================================================
def fig_ring_dataflow(out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    style_fig(fig)
    fig.suptitle("Ring AllReduce: Why 2 Links per Chip is a Bottleneck",
                 fontsize=16, fontweight="bold", color=TEXT, y=0.98)

    for ax_idx, (n_chips, title) in enumerate([(8, "Ring 8 (simple)"),
                                                 (16, "Ring 16 (more hops)")]):
        ax = axes[ax_idx]
        G = nx.cycle_graph(n_chips)
        pos = nx.circular_layout(G)

        # Draw edges
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=EDGE_NORMAL, width=2.0, alpha=0.3)

        # Highlight data flow direction on one half
        flow_edges = [(i, (i+1) % n_chips) for i in range(n_chips // 2)]
        nx.draw_networkx_edges(G, pos, edgelist=flow_edges, ax=ax,
                               edge_color=PAL["coral"], width=3.0, alpha=0.7,
                               arrows=True, arrowsize=15, arrowstyle="-|>",
                               connectionstyle="arc3,rad=0.1")
        # Other direction
        flow_edges2 = [((i+1) % n_chips, i) for i in range(n_chips // 2, n_chips)]
        nx.draw_networkx_edges(G, pos, edgelist=flow_edges2, ax=ax,
                               edge_color=PAL["blue"], width=3.0, alpha=0.7,
                               arrows=True, arrowsize=15, arrowstyle="-|>",
                               connectionstyle="arc3,rad=0.1")

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=300, node_color=PAL["gold"],
                               edgecolors="white", linewidths=2.0, alpha=0.95)
        labels = {i: str(i) for i in range(n_chips)}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8, font_weight="bold",
                                font_color=TEXT)

        ax.set_title(title, fontsize=13, fontweight="bold", color=TEXT)
        ax.text(0, -1.4, "Each chip has only 2 links\n→ bottleneck under heavy traffic",
                ha="center", fontsize=10, color="#888888", style="italic")
        ax.axis("off")
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.6, 1.5)

    # Legend
    cw = mpatches.Patch(color=PAL["coral"], label="Clockwise data flow")
    ccw = mpatches.Patch(color=PAL["blue"], label="Counter-clockwise data flow")
    fig.legend(handles=[cw, ccw], loc="lower center", ncol=2,
               fontsize=11, frameon=True, facecolor="white", edgecolor="#DDDDDD")

    plt.tight_layout(rect=[0, 0.06, 1, 0.92])
    path = out_dir / "ring_dataflow.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


# ===========================================================================
# Figure 5: Torus aspect ratio shapes
# ===========================================================================
def fig_torus_aspects(out_dir):
    fig = plt.figure(figsize=(18, 5.5))
    style_fig(fig)
    fig.suptitle("Torus Aspect Ratios: Same 64 Chips, Different Shapes",
                 fontsize=16, fontweight="bold", color=TEXT, y=0.98)

    configs = [
        ("4×4×4 (cube)\nDiameter = 6", (4, 4, 4), PAL["blue"]),
        ("8×2×4\nDiameter = 7", (4, 2, 8), PAL["green"]),
        ("16×2×2\nDiameter = 10", (2, 2, 16), PAL["purple"]),
    ]

    for idx, (title, dims, color) in enumerate(configs):
        ax = fig.add_subplot(1, 3, idx + 1, projection="3d")
        nodes = grid_nodes_3d(dims)
        mesh_e = mesh_edges(dims)
        wrap_e = torus_extra_edges(dims)
        all_e = mesh_e + wrap_e
        draw_3d_topology(ax, nodes, all_e, title, color,
                         wrap_color=PAL["coral"], node_size=80,
                         elev=20, azim=130)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    path = out_dir / "torus_aspect_ratios.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


# ===========================================================================
# Figure 6: All five proposed topologies compared
# ===========================================================================
def fig_proposed_topologies(out_dir):
    """Side-by-side comparison of all 5 topologies: Ring, Mesh, Torus, 6D Hypercube, Circulant."""
    fig = plt.figure(figsize=(24, 10))
    style_fig(fig)
    fig.suptitle("All Network Topologies Compared (64 chips, 6 links/chip target)",
                 fontsize=18, fontweight="bold", color=TEXT, y=0.98)

    n = 64

    # ---- 1. Ring 64 (circular layout) ----
    ax1 = fig.add_subplot(2, 3, 1)
    G = nx.cycle_graph(n)
    pos = nx.circular_layout(G)
    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color=EDGE_NORMAL, width=0.8, alpha=0.4)
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=30, node_color=PAL["gold"],
                           edgecolors="white", linewidths=0.5, alpha=0.9)
    ax1.set_title("Ring 64\n2 links/chip  |  diam=32", fontsize=12,
                  fontweight="bold", color=TEXT)
    ax1.axis("off")

    # ---- 2. Mesh 4×4×4 (3D) ----
    ax2 = fig.add_subplot(2, 3, 2, projection="3d")
    dims = (4, 4, 4)
    nodes = grid_nodes_3d(dims)
    edges = mesh_edges(dims)
    draw_3d_topology(ax2, nodes, edges,
                     "Mesh 4×4×4\n3–6 links/chip  |  diam=9",
                     PAL["coral"], node_size=60)

    # ---- 3. Torus 4×4×4 (3D with wraparound) ----
    ax3 = fig.add_subplot(2, 3, 3, projection="3d")
    wrap_e = torus_extra_edges(dims)
    draw_3d_topology(ax3, nodes, edges + wrap_e,
                     "Torus 4×4×4\n6 links/chip  |  diam=6",
                     PAL["blue"], wrap_color=PAL["coral"], node_size=60)

    # ---- 4. 6D Hypercube (8×8 grid layout by bit address) ----
    ax4 = fig.add_subplot(2, 3, 4)
    _draw_hypercube_2d(ax4, n=64, ndim=6)
    ax4.set_title("6D Hypercube (2⁶)\n6 links/chip  |  diam=6",
                  fontsize=12, fontweight="bold", color=TEXT)
    ax4.axis("off")
    ax4.set_aspect("equal")

    # ---- 5. Circulant C(64, {1,5,17}) (circular with skip arcs) ----
    ax5 = fig.add_subplot(2, 3, 5)
    _draw_circulant_2d(ax5, n=64, generators=(1, 5, 17))
    ax5.set_title("Circulant C(64, {1,5,17})\n6 links/chip  |  diam=4",
                  fontsize=12, fontweight="bold", color=TEXT)
    ax5.axis("off")
    ax5.set_aspect("equal")

    # ---- 6. Legend / property comparison ----
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    table_data = [
        ["Topology", "Links", "Diam", "AvgHop", "AllReduce\n1 GB"],
        ["Ring 64", "2", "32", "16.25", "slow"],
        ["Mesh 4×4×4", "3–6", "9", "3.81", "33.3 ms"],
        ["Torus 4×4×4", "6", "6", "3.05", "33.3 ms"],
        ["6D Hypercube", "6", "6", "3.05", "22.2 ms"],
        ["Circulant", "6", "4", "2.73", "14.6 ms"],
    ]
    colors_table = ["white", PAL["gold"], PAL["coral"], PAL["blue"],
                    PAL["green"], PAL["lilac"]]
    table = ax6.table(cellText=table_data, loc="center", cellLoc="center",
                      colWidths=[0.28, 0.12, 0.12, 0.15, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for i, row_color in enumerate(colors_table):
        for j in range(5):
            cell = table[i, j]
            if i == 0:
                cell.set_facecolor("#E8E8E8")
                cell.set_text_props(fontweight="bold")
            else:
                cell.set_facecolor(row_color)
                cell.set_alpha(0.3)
            cell.set_edgecolor("#DDDDDD")
    # Highlight best
    for j in range(5):
        table[5, j].set_text_props(fontweight="bold")
    ax6.set_title("Property Comparison", fontsize=12, fontweight="bold", color=TEXT)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    path = out_dir / "all_topologies_compared.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


def _draw_hypercube_2d(ax, n=64, ndim=6):
    """Draw a 6D hypercube using an 8×8 grid layout (3 bits for row, 3 for col)."""
    # Map each node's 6-bit address to a 2D grid position
    # Use Gray code ordering so adjacent nodes in the grid differ by 1 bit
    def gray(k):
        return k ^ (k >> 1)

    rows = 2 ** (ndim // 2)  # 8
    cols = 2 ** (ndim - ndim // 2)  # 8
    gray_row = [gray(i) for i in range(rows)]
    gray_col = [gray(i) for i in range(cols)]

    pos = {}
    for node in range(n):
        row_bits = node >> (ndim // 2)
        col_bits = node & ((1 << (ndim // 2)) - 1)
        # Find gray-code position
        r = gray_row.index(row_bits) if row_bits in gray_row else row_bits
        c = gray_col.index(col_bits) if col_bits in gray_col else col_bits
        pos[node] = (c, rows - 1 - r)

    # Draw edges colored by dimension
    dim_colors = [PAL["blue"], PAL["coral"], PAL["green"],
                  PAL["purple"], PAL["gold"], PAL["pink"]]

    for dim in range(ndim):
        for node in range(n):
            neighbor = node ^ (1 << dim)
            if neighbor > node:  # avoid double-drawing
                p0 = pos[node]
                p1 = pos[neighbor]
                ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                        color=dim_colors[dim % len(dim_colors)],
                        linewidth=0.6, alpha=0.35, zorder=1)

    # Draw nodes
    xs = [pos[i][0] for i in range(n)]
    ys = [pos[i][1] for i in range(n)]
    ax.scatter(xs, ys, s=40, c=PAL["green"], edgecolors="white",
               linewidths=0.8, alpha=0.95, zorder=5)

    # Label corners
    for node in [0, 7, 56, 63]:
        ax.annotate(f"{node:06b}", pos[node], fontsize=6, ha="center",
                    va="bottom", color=TEXT, textcoords="offset points",
                    xytext=(0, 5))


def _draw_circulant_2d(ax, n=64, generators=(1, 5, 17)):
    """Draw a circulant graph with arcs colored by generator."""
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    # Rotate so node 0 is at top
    theta = theta + np.pi / 2
    pos = {i: (np.cos(theta[i]), np.sin(theta[i])) for i in range(n)}

    gen_colors = [PAL["blue"], PAL["coral"], PAL["purple"]]
    gen_labels = []

    for gi, g in enumerate(generators):
        color = gen_colors[gi % len(gen_colors)]
        gen_labels.append((g, color))
        for i in range(n):
            j = (i + g) % n
            p0 = pos[i]
            p1 = pos[j]
            # Determine curvature based on arc distance
            arc_dist = min(abs(j - i), n - abs(j - i))
            rad = 0.0 if arc_dist <= 2 else 0.15 * (arc_dist / n)
            ax.annotate("", xy=p1, xytext=p0,
                        arrowprops=dict(arrowstyle="-", color=color,
                                        linewidth=0.7, alpha=0.3,
                                        connectionstyle=f"arc3,rad={rad}"),
                        zorder=1)

    # Draw nodes
    xs = [pos[i][0] for i in range(n)]
    ys = [pos[i][1] for i in range(n)]
    ax.scatter(xs, ys, s=30, c=PAL["lilac"], edgecolors="white",
               linewidths=0.8, alpha=0.95, zorder=5)

    # Label a few nodes
    for node in [0, 16, 32, 48]:
        ax.annotate(str(node), pos[node], fontsize=7, ha="center", va="center",
                    fontweight="bold", color=TEXT, zorder=6)

    # Generator legend
    for gi, (g, color) in enumerate(gen_labels):
        ax.plot([], [], color=color, linewidth=2, alpha=0.7, label=f"±{g}")
    ax.legend(loc="lower center", fontsize=9, frameon=True, facecolor="white",
              edgecolor="#DDDDDD", ncol=3, title="Generators", title_fontsize=9,
              bbox_to_anchor=(0.5, -0.08))


# ===========================================================================
# Figure 7: Circulant deep-dive — Hamiltonian decomposition
# ===========================================================================
def fig_circulant_detail(out_dir):
    """Show the 3 edge-disjoint Hamiltonian cycles in the Circulant graph."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    style_fig(fig)
    fig.suptitle("Circulant C(64, {1, 5, 17}): Three Edge-Disjoint Hamiltonian Cycles",
                 fontsize=16, fontweight="bold", color=TEXT, y=0.98)

    n = 64
    generators = (1, 5, 17)
    gen_colors = [PAL["blue"], PAL["coral"], PAL["purple"]]

    theta = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
    pos = {i: (np.cos(theta[i]), np.sin(theta[i])) for i in range(n)}

    for gi, (g, color) in enumerate(zip(generators, gen_colors)):
        ax = axes[gi]

        # Draw the Hamiltonian cycle: 0 -> g -> 2g -> ... -> 0 (mod n)
        cycle = [(i * g) % n for i in range(n)]
        cycle.append(cycle[0])  # close the cycle

        # Draw cycle edges
        for k in range(n):
            src, dst = cycle[k], cycle[k + 1]
            p0 = pos[src]
            p1 = pos[dst]
            arc_dist = min(abs(dst - src), n - abs(dst - src))
            rad = 0.0 if arc_dist <= 2 else 0.2 * (arc_dist / n)
            ax.annotate("", xy=p1, xytext=p0,
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        linewidth=1.2, alpha=0.5,
                                        connectionstyle=f"arc3,rad={rad}"),
                        zorder=2)

        # Draw nodes
        xs = [pos[i][0] for i in range(n)]
        ys = [pos[i][1] for i in range(n)]
        ax.scatter(xs, ys, s=25, c=color, edgecolors="white",
                   linewidths=0.6, alpha=0.9, zorder=5)

        # Highlight the first few steps
        for k in range(min(4, n)):
            node = cycle[k]
            ax.annotate(str(node), pos[node], fontsize=7, ha="center", va="center",
                        fontweight="bold", color="white", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", facecolor=color,
                                  edgecolor="white", alpha=0.9))

        ax.set_title(f"Generator ±{g}\nCycle: 0 → {g} → {(2*g)%n} → {(3*g)%n} → ...",
                     fontsize=11, fontweight="bold", color=TEXT)
        ax.axis("off")
        ax.set_aspect("equal")
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)

    fig.text(0.5, 0.02,
             "AllReduce splits data into 3 equal parts — each part flows through one cycle independently (no link contention)",
             ha="center", fontsize=11, color="#888888", style="italic")

    plt.tight_layout(rect=[0, 0.06, 1, 0.92])
    path = out_dir / "circulant_hamiltonian_cycles.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


# ===========================================================================
# Figure 8: 6D Hypercube detail — dimension coloring
# ===========================================================================
def fig_hypercube_detail(out_dir):
    """Show the 6D Hypercube structure with edges colored by dimension."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    style_fig(fig)
    fig.suptitle("6D Hypercube (2⁶ = 64 nodes): Each Edge Flips One Bit",
                 fontsize=16, fontweight="bold", color=TEXT, y=0.98)

    n = 64
    ndim = 6

    # --- Left: Small 4D hypercube (16 nodes) for clarity ---
    n_small = 16
    ndim_small = 4
    theta = np.linspace(0, 2 * np.pi, n_small, endpoint=False) + np.pi / 2
    pos_small = {}
    # Use a layout that puts Hamming-close nodes nearby
    # 4D hypercube: outer ring = first 8, inner ring = next 8
    for i in range(n_small):
        hamming_weight = bin(i).count('1')
        angle = theta[i]
        radius = 1.0 - 0.15 * hamming_weight
        pos_small[i] = (radius * np.cos(angle), radius * np.sin(angle))

    dim_colors = [PAL["blue"], PAL["coral"], PAL["green"], PAL["purple"],
                  PAL["gold"], PAL["pink"]]
    dim_labels = [f"bit {d}" for d in range(ndim_small)]

    for dim in range(ndim_small):
        edges = []
        for node in range(n_small):
            neighbor = node ^ (1 << dim)
            if neighbor > node:
                edges.append((node, neighbor))
        for src, dst in edges:
            p0, p1 = pos_small[src], pos_small[dst]
            ax1.plot([p0[0], p1[0]], [p0[1], p1[1]],
                     color=dim_colors[dim], linewidth=1.5, alpha=0.4, zorder=1)

    xs = [pos_small[i][0] for i in range(n_small)]
    ys = [pos_small[i][1] for i in range(n_small)]
    ax1.scatter(xs, ys, s=120, c=PAL["sky"], edgecolors="white",
                linewidths=1.5, zorder=5)
    for i in range(n_small):
        ax1.annotate(f"{i:04b}", pos_small[i], fontsize=6.5, ha="center", va="center",
                     fontweight="bold", color=TEXT, zorder=6)

    for dim in range(ndim_small):
        ax1.plot([], [], color=dim_colors[dim], linewidth=2.5, label=dim_labels[dim])
    ax1.legend(loc="lower center", fontsize=9, frameon=True, facecolor="white",
               edgecolor="#DDDDDD", ncol=4, bbox_to_anchor=(0.5, -0.05))
    ax1.set_title("4D Hypercube (16 nodes)\nfor visual clarity", fontsize=12,
                  fontweight="bold", color=TEXT)
    ax1.axis("off")
    ax1.set_aspect("equal")

    # --- Right: Full 6D hypercube with 8×8 grid layout ---
    _draw_hypercube_2d(ax2, n=64, ndim=6)

    # Add dimension legend
    for dim in range(6):
        ax2.plot([], [], color=dim_colors[dim], linewidth=2, alpha=0.7,
                 label=f"dim {dim}")
    ax2.legend(loc="lower center", fontsize=8, frameon=True, facecolor="white",
               edgecolor="#DDDDDD", ncol=6, bbox_to_anchor=(0.5, -0.05))
    ax2.set_title("6D Hypercube (64 nodes)\n8×8 Gray-code grid layout",
                  fontsize=12, fontweight="bold", color=TEXT)

    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    path = out_dir / "hypercube_detail.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {path}")


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating topology & workload visualizations in {out_dir}/...")
    fig_all_topologies(out_dir)
    fig_torus_detail(out_dir)
    fig_matrix_workloads(out_dir)
    fig_ring_dataflow(out_dir)
    fig_torus_aspects(out_dir)
    fig_proposed_topologies(out_dir)
    fig_circulant_detail(out_dir)
    fig_hypercube_detail(out_dir)

    print(f"\nDone! {len(list(out_dir.glob('*.png')))} figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
