"""
Network cost model with link-level congestion.

Routes transfers onto physical links, merges loads from concurrent transfers,
and computes cost from the bottleneck link.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from network_topology.topology import Topology


class CollectiveType(Enum):
    BROADCAST = auto()
    ALLREDUCE = auto()
    REDUCE_SCATTER = auto()
    ALLGATHER = auto()
    POINT_TO_POINT = auto()


@dataclass
class NetworkTransfer:
    tensor_name: str
    data_bytes: float
    collective_type: CollectiveType
    src_chip: Optional[int] = None
    dst_chips: Optional[list[int]] = None
    participating_chips: Optional[list[int]] = None


@dataclass
class NetworkCostResult:
    total_energy: float
    total_latency: float
    per_transfer: list[dict]
    energy_per_network_access: float
    latency_per_network_access: float
    total_network_bytes: float

    def summary(self) -> str:
        lines = [
            f"Energy: {self.total_energy:.4e} J, Latency: {self.total_latency:.4e} s, "
            f"Bytes: {self.total_network_bytes:.2e}",
        ]
        for t in self.per_transfer:
            lines.append(
                f"  {t['tensor']:>12s} {t['collective']:>15s} "
                f"E={t['energy']:.4e} L={t['latency']:.4e} {t['data_bytes']:.2e}B"
            )
        return "\n".join(lines)


def _get_transfer_link_loads(topology, transfer):
    all_chips = list(range(topology.num_chips))
    ct = transfer.collective_type

    if ct == CollectiveType.BROADCAST:
        src = transfer.src_chip or 0
        dst = transfer.dst_chips or [i for i in all_chips if i != src]
        return topology._broadcast_link_loads(transfer.data_bytes, src, dst)
    elif ct == CollectiveType.ALLREDUCE:
        return topology._allreduce_link_loads(transfer.data_bytes, transfer.participating_chips or all_chips)
    elif ct in (CollectiveType.REDUCE_SCATTER, CollectiveType.ALLGATHER):
        loads = topology._allreduce_link_loads(transfer.data_bytes, transfer.participating_chips or all_chips)
        return {k: v / 2 for k, v in loads.items()}
    elif ct == CollectiveType.POINT_TO_POINT:
        src = transfer.src_chip or 0
        dst = transfer.dst_chips[0] if transfer.dst_chips else 1
        return topology._point_to_point_link_loads(transfer.data_bytes, src, dst)
    else:
        raise ValueError(f"Unknown collective type: {ct}")


def compute_network_cost(topology: Topology, transfers: list[NetworkTransfer]) -> NetworkCostResult:
    """
    Compute congested network cost for concurrent transfers.

    Each transfer is routed onto physical links. All link loads are merged.
    The bottleneck link determines overall latency. Energy = total bit-hops.
    """
    if not transfers:
        return NetworkCostResult(0, 0, [], 0, 0, 0)

    per_transfer_loads = []
    per_transfer = []
    total_bytes = 0.0

    for transfer in transfers:
        loads = _get_transfer_link_loads(topology, transfer)
        per_transfer_loads.append(loads)
        indiv_energy, indiv_latency = topology._cost_from_link_loads(loads)
        per_transfer.append({
            "tensor": transfer.tensor_name,
            "collective": transfer.collective_type.name,
            "energy": indiv_energy, "latency": indiv_latency,
            "data_bytes": transfer.data_bytes,
        })
        total_bytes += transfer.data_bytes

    total_energy, total_latency = topology.compute_congested_cost(per_transfer_loads)

    epb = total_energy / total_bytes if total_bytes > 0 else 0
    lpb = total_latency / total_bytes if total_bytes > 0 else 0

    return NetworkCostResult(total_energy, total_latency, per_transfer, epb, lpb, total_bytes)
