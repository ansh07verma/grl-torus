"""
SimPy discrete-event simulator for 2D torus optical network.

Orchestrates packet lifecycle: inject → route (per-hop) → queue → transmit → receive.
Collects comprehensive metrics for analysis.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import simpy

from src.routers.base import BaseRouter
from src.sim.link import TorusLink
from src.sim.packet import Packet
from src.sim.torus_graph import TorusGraph
from src.sim.traffic import TrafficGenerator
from src.utils.logging import get_logger

logger = get_logger("sim.simulator")


@dataclass
class SimulationResults:
    """Container for simulation metrics from a single run.

    Attributes:
        packets: All packets (delivered, dropped, and in-flight).
        duration_ns: Total simulation duration in ns.
        warmup_ns: Warmup period (metrics from before this are discarded).
    """

    packets: List[Packet] = field(default_factory=list)
    duration_ns: float = 0.0
    warmup_ns: float = 0.0

    # Link utilisation snapshots
    link_utilisations: Dict[Tuple[Tuple[int, int], Tuple[int, int]], float] = field(
        default_factory=dict
    )

    # Timing
    wall_time_seconds: float = 0.0

    @property
    def _post_warmup_packets(self) -> List[Packet]:
        """Packets created after the warmup period."""
        return [p for p in self.packets if p.creation_time >= self.warmup_ns]

    @property
    def delivered_packets(self) -> List[Packet]:
        """Successfully delivered packets (post-warmup)."""
        return [p for p in self._post_warmup_packets if p.delivered]

    @property
    def dropped_packets(self) -> List[Packet]:
        """Dropped packets (post-warmup)."""
        return [p for p in self._post_warmup_packets if p.dropped]

    def avg_latency(self) -> float:
        """Average end-to-end latency in ns."""
        delivered = self.delivered_packets
        if not delivered:
            return float("inf")
        latencies = [p.total_latency for p in delivered if p.total_latency is not None]
        return float(np.mean(latencies)) if latencies else float("inf")

    def p95_latency(self) -> float:
        """95th percentile latency in ns."""
        delivered = self.delivered_packets
        if not delivered:
            return float("inf")
        latencies = [p.total_latency for p in delivered if p.total_latency is not None]
        return float(np.percentile(latencies, 95)) if latencies else float("inf")

    def throughput(self) -> float:
        """Throughput: delivered packets per nanosecond."""
        effective_duration = self.duration_ns - self.warmup_ns
        if effective_duration <= 0:
            return 0.0
        return len(self.delivered_packets) / effective_duration

    def drop_rate(self) -> float:
        """Packet drop rate as a fraction [0, 1]."""
        total = len(self._post_warmup_packets)
        if total == 0:
            return 0.0
        return len(self.dropped_packets) / total

    def mean_utilisation(self) -> float:
        """Mean link utilisation across all links."""
        if not self.link_utilisations:
            return 0.0
        return float(np.mean(list(self.link_utilisations.values())))

    def max_utilisation(self) -> float:
        """Maximum link utilisation."""
        if not self.link_utilisations:
            return 0.0
        return float(np.max(list(self.link_utilisations.values())))

    def avg_hop_count(self) -> float:
        """Average hop count for delivered packets."""
        delivered = self.delivered_packets
        if not delivered:
            return 0.0
        return float(np.mean([p.hop_count for p in delivered]))

    def to_dict(self) -> Dict[str, Any]:
        """Export metrics as flat dictionary."""
        return {
            "avg_latency_ns": self.avg_latency(),
            "p95_latency_ns": self.p95_latency(),
            "throughput_pps": self.throughput(),
            "drop_rate": self.drop_rate(),
            "mean_utilisation": self.mean_utilisation(),
            "max_utilisation": self.max_utilisation(),
            "avg_hop_count": self.avg_hop_count(),
            "total_packets": len(self._post_warmup_packets),
            "delivered_packets": len(self.delivered_packets),
            "dropped_packets": len(self.dropped_packets),
            "wall_time_s": self.wall_time_seconds,
        }


class TorusSimulator:
    """SimPy discrete-event simulator for 2D torus network.

    Orchestrates the full packet lifecycle and collects metrics.

    Args:
        torus: The TorusGraph instance.
        router: The routing algorithm to use.
        duration_ns: Simulation duration in nanoseconds.
        warmup_ns: Warmup period — metrics before this time are discarded.
        max_hops: Maximum hops before a packet is dropped (loop prevention).
        queue_timeout_ns: Max time a packet can wait in a queue before being dropped.
    """

    def __init__(
        self,
        torus: TorusGraph,
        router: BaseRouter,
        duration_ns: float = 1_000_000,
        warmup_ns: float = 100_000,
        max_hops: int = 0,
        queue_timeout_ns: float = 50_000,
    ):
        self.torus = torus
        self.router = router
        self.duration_ns = duration_ns
        self.warmup_ns = warmup_ns
        self.queue_timeout_ns = queue_timeout_ns

        # Max hops = 4*N for safety (well above theoretical max of N)
        self.max_hops = max_hops if max_hops > 0 else 4 * torus.n

        # SimPy environment
        self.env = simpy.Environment()

        # Packet tracking
        self._all_packets: List[Packet] = []
        self._active_packets: int = 0

        # Graph state cache (invalidated each tick)
        self._cached_state: Optional[Dict[str, Any]] = None
        self._cached_state_time: float = -1.0

    def _get_graph_state(self) -> Dict[str, Any]:
        """Get cached graph state snapshot (refreshed once per simulation time unit)."""
        if self._cached_state_time != self.env.now:
            self._cached_state = self.torus.get_graph_state()
            self._cached_state_time = self.env.now
        return self._cached_state  # type: ignore

    def _packet_lifecycle(self, packet: Packet) -> Any:
        """SimPy process managing a single packet's journey through the torus.

        Stages:
            1. Inject — packet arrives at source
            2. Route — per-hop next-hop decision
            3. Queue — wait for output queue slot
            4. Transmit — traverse the link
            5. Receive — arrive at next node; repeat from (2) if not at destination
        """
        self._active_packets += 1
        current = packet.src
        packet.record_hop(current)

        try:
            while current != packet.dst:
                # Check simulation time limit
                if self.env.now >= self.duration_ns:
                    if not packet.delivered and not packet.dropped:
                        packet.mark_dropped("simulation_ended")
                    break

                # Check hop limit (loop prevention)
                if packet.hop_count >= self.max_hops:
                    packet.mark_dropped("max_hops_exceeded")
                    self.torus.nodes[current].packets_dropped += 1
                    break

                # Update node load
                self.torus.nodes[current].update_load()

                # Get routing decision
                graph_state = self._get_graph_state()
                next_hop = self.router.route(packet, current, graph_state, self.torus)

                if next_hop is None:
                    # Hold action — packet stays in current node
                    packet.consecutive_holds += 1
                    if packet.consecutive_holds > 3:
                        # Forced deflection after 3 consecutive holds
                        active_neighbors = self.torus.get_active_neighbors(current)
                        if active_neighbors:
                            # Pick random active neighbor
                            directions = list(active_neighbors.keys())
                            idx = hash(packet.id + int(self.env.now)) % len(directions)
                            next_hop = active_neighbors[directions[idx]]
                            packet.consecutive_holds = 0
                        else:
                            # Truly stuck — drop
                            packet.mark_dropped("no_active_neighbors")
                            self.torus.nodes[current].packets_dropped += 1
                            break
                    else:
                        # Wait one tick and retry
                        yield self.env.timeout(1)
                        continue
                else:
                    packet.consecutive_holds = 0

                # Get the link and direction
                direction = self.torus.get_direction(current, next_hop)
                if direction is None:
                    packet.mark_dropped("invalid_next_hop")
                    break

                link = self.torus.get_link(current, next_hop)
                if link is None or link.is_failed:
                    packet.mark_dropped("link_failed")
                    self.torus.nodes[current].packets_dropped += 1
                    break

                # Queue phase — try to enqueue in the output queue
                node_obj = self.torus.nodes[current]
                queue = node_obj.get_queue(direction)

                if queue.is_full:
                    # Buffer overflow — drop packet
                    packet.mark_dropped("buffer_overflow")
                    node_obj.packets_dropped += 1
                    break

                # Enqueue (instantaneous in this model)
                queue.enqueue(packet)

                # Transmit phase — wait for transmission + propagation delay
                delay = link.transmit(packet.payload_size)
                packet.transmission_time += delay
                yield self.env.timeout(delay)

                # Dequeue (packet leaves the queue after transmission)
                queue.dequeue()

                # Arrive at next hop
                current = next_hop
                packet.record_hop(current)
                self.torus.nodes[current].packets_processed += 1

            # Delivery
            if current == packet.dst and not packet.dropped:
                packet.mark_delivered(self.env.now)

        except simpy.Interrupt:
            if not packet.delivered and not packet.dropped:
                packet.mark_dropped("interrupted")

        finally:
            self._active_packets -= 1

    def _packet_injector(self, traffic_gen: TrafficGenerator) -> Any:
        """SimPy process that injects packets from the traffic generator."""
        while self.env.now < self.duration_ns:
            # Poisson inter-arrival time
            inter_arrival = traffic_gen._inter_arrival_time()
            yield self.env.timeout(inter_arrival)

            # Stop generating after simulation duration
            if self.env.now >= self.duration_ns:
                break

            # Generate packet
            src, dst = traffic_gen._generate_pair()
            traffic_gen._packet_counter += 1

            packet = Packet(
                id=traffic_gen._packet_counter,
                src=src,
                dst=dst,
                creation_time=self.env.now,
            )

            self._all_packets.append(packet)
            self.torus.nodes[src].packets_generated += 1

            # Start packet lifecycle process
            self.env.process(self._packet_lifecycle(packet))

    def run(self, traffic_gen: TrafficGenerator) -> SimulationResults:
        """Run the simulation.

        Args:
            traffic_gen: Traffic generator producing packets.

        Returns:
            SimulationResults with all collected metrics.
        """
        start_wall = time.time()

        logger.info(
            f"Starting simulation: {self.torus.n}x{self.torus.n} torus, "
            f"router={self.router.name}, duration={self.duration_ns}ns"
        )

        # Inject failures for fault traffic pattern
        traffic_gen.inject_failures(self.torus)

        # Start packet injection process
        self.env.process(self._packet_injector(traffic_gen))

        # Run simulation
        # Run simulation — add a small buffer to let in-flight packets complete
        self.env.run(until=self.duration_ns + self.torus.n * 100)

        wall_time = time.time() - start_wall

        # Collect link utilisations
        link_utils = {}
        for key, link in self.torus.links.items():
            link_utils[key] = link.utilisation

        results = SimulationResults(
            packets=self._all_packets,
            duration_ns=self.duration_ns,
            warmup_ns=self.warmup_ns,
            link_utilisations=link_utils,
            wall_time_seconds=wall_time,
        )

        metrics = results.to_dict()
        logger.info(
            f"Simulation complete in {wall_time:.2f}s: "
            f"delivered={metrics['delivered_packets']}, "
            f"dropped={metrics['dropped_packets']}, "
            f"avg_latency={metrics['avg_latency_ns']:.1f}ns, "
            f"throughput={metrics['throughput_pps']:.6f} pkt/ns"
        )

        return results

    def reset(self) -> None:
        """Reset simulator for a new run."""
        self.env = simpy.Environment()
        self._all_packets.clear()
        self._active_packets = 0
        self._cached_state = None
        self._cached_state_time = -1.0
        self.torus.reset()
        self.router.reset()
