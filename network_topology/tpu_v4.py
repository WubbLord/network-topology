"""
TPU v4 hardware constants and topology factory.

Sources:
    [1] Jouppi et al., "TPU v4: An Optically Reconfigurable Supercomputer", ISCA 2023
    [2] Google Cloud TPU v4 docs (docs.cloud.google.com/tpu/docs/v4)
    [3] JAX Scaling Book (jax-ml.github.io/scaling-book/tpus/)
    [4] Cariou et al., "Mission Apollo: Landing OCS at Datacenter Scale", 2022
"""

from network_topology.topology import Torus3D, Mesh3D

# --- Chip-level ---
CHIP_PEAK_TFLOPS_BF16 = 275.0          # [2]
CHIP_HBM_CAPACITY_BYTES = 32 * 1024**3 # 32 GiB HBM2 [2]
CHIP_HBM_BANDWIDTH = 1200e9            # 1.2 TB/s [2]
CHIP_TDP_WATTS = 170.0                 # [1]

# --- ICI (Inter-Chip Interconnect) ---
ICI_LINKS_PER_CHIP = 6                 # 2 per dimension in 3D [1][2][3]
ICI_LINK_BW_UNIDIR = 45e9              # 45 GB/s per direction [3]
ICI_LINK_BW_BIDIR = 90e9               # full-duplex [3]
ICI_PER_HOP_LATENCY = 500e-9           # ~500 ns (estimated, not published)
ICI_ENERGY_PER_BIT_PER_HOP = 4e-12     # ~4 pJ/bit (estimated from <5% system power [1])

# --- Pod-level ---
POD_NUM_CHIPS = 4096                    # [1][2]
POD_ALLREDUCE_BW = 1.1e15              # 1.1 PB/s [2]
POD_BISECTION_BW = 24e12               # 24 TB/s [2]
CUBE_DIMS = (4, 4, 4)                  # 64 chips per cube, copper ICI [1][4]


def make_tpu_v4_topology(dims=CUBE_DIMS, force_mesh=False):
    """
    Create a TPU v4 topology. Auto-selects torus (dims >= 4) or mesh.
    Use force_mesh=True to compare mesh vs torus on the same dimensions.
    """
    hw = dict(
        link_bandwidth=ICI_LINK_BW_UNIDIR,
        energy_per_bit_per_hop=ICI_ENERGY_PER_BIT_PER_HOP,
        per_hop_latency=ICI_PER_HOP_LATENCY,
        full_duplex=True,
    )
    use_torus = not force_mesh and all(d >= 4 for d in dims)
    cls = Torus3D if use_torus else Mesh3D
    return cls(dims=dims, **hw)
