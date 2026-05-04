"""
edmonds_karp.py
───────────────
The Edmonds-Karp variant of Ford-Fulkerson for the maximum-flow /
minimum-cut problem on a directed network.

Edmonds-Karp is the Ford-Fulkerson labelling algorithm with one
specific scan rule: the labelled-but-unscanned vertex queue is
processed in FIFO order.  This is equivalent to a breadth-first
search of the residual network, so each augmenting path it finds
is a shortest augmenting path (fewest residual arcs).  As a result:

    Theorem (Edmonds & Karp, 1972).
    Edmonds-Karp terminates after at most O(|V|·|E|) augmentations
    and runs in O(|V|·|E|^2) time.

This file is both:
  * a teaching tool — set ``verbose=True`` and every label, queue
    event, traceback step, residual update, and cumulative flow
    value is printed in the (±i, \u03B4) notation Dr. Balasundaram's
    lecture slides use; and
  * a reusable library — set ``verbose=False`` and inspect the
    structured ``MaxFlowResult`` returned by ``solve()``.

Quickstart
───────────
```
    from edmonds_karp import EdmondsKarp

    capacity = {
        1: {2: 20, 3: 13},
        2: {4:  8, 5:  6},
        3: {5: 12, 6: 14},
        4: {5: 11, 7: 21},
        5: {6:  3, 7: 10},
        6: {7:  9},
        7: {},
    }
    result = EdmondsKarp(capacity, source=1, sink=7).solve()
    print(result.summary())
```
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional, Set, Tuple


# ──────────────────────────────────────────────────────────────────────
# Type aliases
# ──────────────────────────────────────────────────────────────────────
Vertex   = Hashable
Capacity = Dict[Vertex, Dict[Vertex, float]]
Label    = Tuple[Optional[Vertex], str, float]   # (predecessor, sign, delta)


# ──────────────────────────────────────────────────────────────────────
# Result types
# ──────────────────────────────────────────────────────────────────────
@dataclass
class IterationLog:
    """Snapshot of a single iteration."""
    iteration:       int
    labels:          Dict[Vertex, Label]
    augmenting_path: Optional[List[Vertex]]    # None ⇒ termination iteration
    bottleneck:      Optional[float]
    cumulative_flow: float
    terminated:      bool


@dataclass
class MaxFlowResult:
    """Output of ``EdmondsKarp.solve()``."""
    max_flow_value: float
    flow:           Dict[Vertex, Dict[Vertex, float]]
    source_side:    Set[Vertex]
    sink_side:      Set[Vertex]
    cut_arcs:       List[Tuple[Vertex, Vertex, float]]   # (u, v, capacity)
    cut_capacity:   float
    iterations:     List[IterationLog] = field(default_factory=list)

    def summary(self) -> str:
        ok = "✓  matches max flow" if self.cut_capacity == self.max_flow_value \
             else "✗  mismatch — bug!"
        lines = [
            "═" * 64,
            "  Max-Flow / Min-Cut Result  (Edmonds-Karp)",
            "═" * 64,
            f"  Max flow value         : {self.max_flow_value}",
            f"  Min cut capacity       : {self.cut_capacity}    {ok}",
            f"  Source-side  S         : {sorted(self.source_side)}",
            f"  Sink-side    T         : {sorted(self.sink_side)}",
            f"  Iterations (incl. last): {len(self.iterations)}",
            "  Cut arcs (S → T):",
        ]
        for u, v, c in self.cut_arcs:
            lines.append(f"    ({u},{v})  capacity {c}")
        if self.flow:
            lines.append("  Final non-zero flows:")
            for u in sorted(self.flow):
                for v in sorted(self.flow[u]):
                    lines.append(f"    f({u},{v}) = {self.flow[u][v]}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Solver
# ──────────────────────────────────────────────────────────────────────

class EdmondsKarp:
    """
    Edmonds-Karp algorithm for maximum flow.

    Parameters
    ----------
    capacity : dict-of-dicts ``{u: {v: cap_uv}}``
        Capacity function of the directed network.  Vertices may be
        any hashable type; absent arcs are treated as zero capacity.
    source, sink : Vertex
        The two distinguished vertices.  Must be distinct.
    verbose : bool, default True
        If True, prints a full step-by-step trace of every iteration.

    Notes
    -----
    Inside each iteration:
      [1] Initialise: label[source] = (-, \u221E);  enqueue source.
      [2] BFS scan: pop FIFO; for each residual neighbour with r > 0,
          label it (±predecessor, min(\u03B4_pred, r)) and enqueue it;
          stop as soon as the sink is labelled.
      [3] Trace back from sink to source via the predecessor pointers
          to recover the shortest augmenting path.
      [4] Augment by \u03B4 = label[sink][2] (forward arcs add \u03B4, backward
          arcs subtract \u03B4 from the original arc's flow).
      [5] Update the residual network (implicit in flow update).
      [6] Add \u03B4 to the cumulative flow and clear all labels.

    On termination (sink not reached during BFS), the labelled set is
    the source side of a minimum cut, and its capacity equals the
    maximum flow.
    """

    def __init__(
        self,
        capacity: Capacity,
        source:   Vertex,
        sink:     Vertex,
        *,
        verbose:  bool = True,
    ):
        if source == sink:
            raise ValueError("source and sink must be distinct vertices")

        self.capacity = capacity
        self.source   = source
        self.sink     = sink
        self.verbose  = verbose

        # Vertex set = union of all keys and values
        verts: Set[Vertex] = set(capacity.keys())
        for u in capacity:
            verts |= set(capacity[u].keys())
        self.vertices = sorted(verts)

        # Mutable algorithm state
        self.flow: Dict[Vertex, Dict[Vertex, float]] = (
            defaultdict(lambda: defaultdict(int))
        )
        self._cumulative: float = 0
        self._iteration_logs: List[IterationLog] = []

    # ── Residual graph helpers ────────────────────────────────────
    def residual(self, u: Vertex, v: Vertex) -> float:
        """Residual capacity r(u,v) = forward slack + reverse flow."""
        cap_uv = self.capacity.get(u, {}).get(v, 0)
        return (cap_uv - self.flow[u][v]) + self.flow[v][u]

    def arc_sign(self, u: Vertex, v: Vertex) -> str:
        """'+' if (u,v) is a forward residual arc, '-' if backward."""
        cap_uv = self.capacity.get(u, {}).get(v, 0)
        return '+' if (cap_uv > 0 and self.flow[u][v] < cap_uv) else '-'

    def residual_candidates(self, u: Vertex) -> List[Vertex]:
        """All v that could currently be residual neighbours of u
        (in sorted-vertex order, matching the textbook 'sorted
        adjacency list' convention)."""
        cands: Set[Vertex] = set(self.capacity.get(u, {}).keys())
        for v in self.vertices:
            if self.flow[v][u] > 0:
                cands.add(v)
        return sorted(cands)

    # ── Pretty printing ───────────────────────────────────────────
    def _fmt_label(self, lab: Label) -> str:
        pred, sign, delta = lab
        if pred is None:
            return "(-, \u221E)"
        d = "\u221E" if delta == float('inf') else str(delta)
        return f"({sign}{pred}, {d})"

    def _print_residual_network(self) -> None:
        print("           Residual capacities (positive only):")
        for u in self.vertices:
            arcs = []
            for v in self.residual_candidates(u):
                r = self.residual(u, v)
                if r > 0:
                    arcs.append(f"({u},{v},{self.arc_sign(u,v)},r={r})")
            if arcs:
                print(f"             {u}: " + ", ".join(arcs))

    def _vprint(self, *args, **kwargs) -> None:
        if self.verbose:
            print(*args, **kwargs)

    def _print_problem(self) -> None:
        if not self.verbose:
            return
        print("Edmonds-Karp Algorithm  (Ford-Fulkerson with FIFO/BFS scan)")
        print(f"Source = {self.source}, Sink = {self.sink}")
        print("\nOriginal arcs (capacities):")
        for u in self.vertices:
            for v, c in self.capacity.get(u, {}).items():
                print(f"  ({u},{v})  capacity {c}")

    # ── Main loop ─────────────────────────────────────────────────
    def solve(self) -> MaxFlowResult:
        self._print_problem()

        iteration = 0
        while True:
            iteration += 1
            self._vprint(f"\n{'='*72}")
            self._vprint(f"  ITERATION {iteration}")
            self._vprint(f"{'='*72}")

            # ── Step 1: Initialise labels and FIFO queue ─────────
            label: Dict[Vertex, Label] = {
                self.source: (None, '-', float('inf'))
            }
            queue: deque = deque([self.source])
            self._vprint(f"\n  [1] Initialise: label[{self.source}] = (-, \u221E);  "
                         f"queue = {list(queue)}")

            # ── Step 2: Breadth-first search (BFS) scan ─────────────────────────────────
            self._vprint(f"\n  [2] BFS (FIFO) scan:")
            sink_labelled = False
            while queue:
                u = queue.popleft()                     # FIFO
                delta_u = label[u][2]
                d_str = "\u221E" if delta_u == float('inf') else str(delta_u)
                cands = self.residual_candidates(u)
                self._vprint(f"      ── pop {u}  (carries \u03B4={d_str});  "
                             f"candidates {cands}")

                for v in cands:
                    r_uv = self.residual(u, v)
                    if r_uv == 0:
                        self._vprint(f"           · {v}: r({u},{v})=0  → skip")
                        continue
                    if v in label:
                        self._vprint(f"           · {v}: r({u},{v})={r_uv}  "
                                     f"but already labelled → skip")
                        continue
                    sign = self.arc_sign(u, v)
                    new_delta = min(delta_u, r_uv)
                    label[v] = (u, sign, new_delta)
                    queue.append(v)
                    self._vprint(
                        f"           · {v}: r({u},{v})={r_uv}  → "
                        f"label[{v}]={self._fmt_label(label[v])};  "
                        f"queue={list(queue)}"
                    )
                    if v == self.sink:
                        sink_labelled = True
                        break
                if sink_labelled:
                    break

            # ── Termination ──────────────────────────────────────
            if not sink_labelled:
                self._iteration_logs.append(IterationLog(
                    iteration=iteration,
                    labels=dict(label),
                    augmenting_path=None,
                    bottleneck=None,
                    cumulative_flow=self._cumulative,
                    terminated=True,
                ))
                self._vprint(f"\n      Queue exhausted; sink {self.sink} "
                             f"not labelled.")
                self._vprint(f"      ALGORITHM TERMINATES.")
                return self._build_result(label)

            # ── Step 3: Traceback ────────────────────────────────
            self._vprint(f"\n  [3] Traceback from sink:")
            path: List[Tuple[Vertex, Vertex, str]] = []
            v = self.sink
            while v != self.source:
                pred, sign, _ = label[v]
                path.append((pred, v, sign))
                self._vprint(f"      label[{v}] = {self._fmt_label(label[v])}"
                             f"  →  predecessor {pred}")
                v = pred
            path.reverse()
            delta = label[self.sink][2]
            path_vertices = [self.source] + [w for _, w, _ in path]
            self._vprint(f"\n      Augmenting path : "
                         f"{' → '.join(map(str, path_vertices))}")
            self._vprint(f"      Bottleneck \u03B4    : {delta}")

            # ── Step 4: Augment ──────────────────────────────────
            self._vprint(f"\n  [4] Augment by \u03B4 = {delta}:")
            for (u, v, sign) in path:
                if sign == '+':
                    self.flow[u][v] += delta
                    self._vprint(
                        f"      ({u},{v}) forward : f({u},{v}) ← "
                        f"{self.flow[u][v] - delta} + {delta} = {self.flow[u][v]}"
                    )
                else:
                    self.flow[v][u] -= delta
                    self._vprint(
                        f"      ({u},{v}) backward: f({v},{u}) ← "
                        f"{self.flow[v][u] + delta} − {delta} = {self.flow[v][u]}"
                    )

            self._cumulative += delta

            # ── Steps 5 & 6: Residual snapshot + cumulative flow ─
            self._iteration_logs.append(IterationLog(
                iteration=iteration,
                labels=dict(label),
                augmenting_path=path_vertices,
                bottleneck=delta,
                cumulative_flow=self._cumulative,
                terminated=False,
            ))
            if self.verbose:
                print(f"\n  [5] Updated residual network:")
                self._print_residual_network()
                print(f"\n  [6] Cumulative flow after iteration "
                      f"{iteration}: {self._cumulative}")

    # ── Result construction ──────────────────────────────────────
    def _build_result(self, final_label: Dict[Vertex, Label]) -> MaxFlowResult:
        S = set(final_label.keys())
        T = set(self.vertices) - S
        cut_arcs = [
            (u, v, self.capacity[u][v])
            for u in S
            for v in self.capacity.get(u, {})
            if v in T
        ]
        cut_cap = sum(c for _, _, c in cut_arcs)

        flow_out: Dict[Vertex, Dict[Vertex, float]] = {}
        for u in self.vertices:
            row = {v: self.flow[u][v]
                   for v in self.capacity.get(u, {})
                   if self.flow[u][v] > 0}
            if row:
                flow_out[u] = row

        result = MaxFlowResult(
            max_flow_value=self._cumulative,
            flow=flow_out,
            source_side=S,
            sink_side=T,
            cut_arcs=cut_arcs,
            cut_capacity=cut_cap,
            iterations=self._iteration_logs,
        )
        if self.verbose:
            print(f"\n{result.summary()}")
        return result