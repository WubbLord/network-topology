"""
Network cost model with per-collective link-level costs.

Each collective is routed onto physical links independently. Per-collective
latency comes from that collective's bottleneck link, and total latency is the
sum across collectives.
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


def _validate_chip_ids(topology: Topology, chips: list[int], field_name: str) -> None:
    invalid = [chip for chip in chips if chip < 0 or chip >= topology.num_chips]
    if invalid:
        raise ValueError(
            f"{field_name} contains chips outside topology range "
            f"0..{topology.num_chips - 1}: {invalid}"
        )


def _get_transfer_link_loads(topology, transfer):
    all_chips = list(range(topology.num_chips))
    ct = transfer.collective_type

    if ct == CollectiveType.BROADCAST:
        src = transfer.src_chip or 0
        dst = transfer.dst_chips or [i for i in all_chips if i != src]
        _validate_chip_ids(topology, [src, *dst], "BROADCAST chips")
        return topology._broadcast_link_loads(transfer.data_bytes, src, dst)
    elif ct == CollectiveType.ALLREDUCE:
        participants = transfer.participating_chips or all_chips
        _validate_chip_ids(topology, participants, "ALLREDUCE participating_chips")
        return topology._allreduce_link_loads(transfer.data_bytes, participants)
    elif ct in (CollectiveType.REDUCE_SCATTER, CollectiveType.ALLGATHER):
        participants = transfer.participating_chips or all_chips
        _validate_chip_ids(topology, participants, f"{ct.name} participating_chips")
        loads = topology._allreduce_link_loads(transfer.data_bytes, participants)
        return {k: v / 2 for k, v in loads.items()}
    elif ct == CollectiveType.POINT_TO_POINT:
        src = transfer.src_chip or 0
        dst = transfer.dst_chips[0] if transfer.dst_chips else 1
        _validate_chip_ids(topology, [src, dst], "POINT_TO_POINT chips")
        return topology._point_to_point_link_loads(transfer.data_bytes, src, dst)
    else:
        raise ValueError(f"Unknown collective type: {ct}")


def compute_network_cost(topology: Topology, transfers: list[NetworkTransfer]) -> NetworkCostResult:
    """
    Compute network cost for a sequential list of collective operations.

    Link loads are not merged across transfers. Each transfer is routed and
    costed independently, then energy and latency are summed across transfers.
    """
    if not transfers:
        return NetworkCostResult(0, 0, [], 0, 0, 0)

    per_transfer = []
    total_bytes = 0.0
    total_energy = 0.0
    total_latency = 0.0

    for transfer in transfers:
        loads = _get_transfer_link_loads(topology, transfer)
        indiv_energy, indiv_latency = topology._cost_from_link_loads(loads)
        per_transfer.append({
            "tensor": transfer.tensor_name,
            "collective": transfer.collective_type.name,
            "energy": indiv_energy, "latency": indiv_latency,
            "data_bytes": transfer.data_bytes,
            "src_chip": transfer.src_chip,
            "dst_chips": transfer.dst_chips,
            "participating_chips": transfer.participating_chips,
            "link_count": len(loads),
            "max_link_load": max(loads.values(), default=0.0),
        })
        total_bytes += transfer.data_bytes
        total_energy += indiv_energy
        total_latency += indiv_latency

    epb = total_energy / total_bytes if total_bytes > 0 else 0
    lpb = total_latency / total_bytes if total_bytes > 0 else 0

    return NetworkCostResult(total_energy, total_latency, per_transfer, epb, lpb, total_bytes)
