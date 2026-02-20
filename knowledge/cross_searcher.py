"""
Cross Searcher
==============
Search utility for the Cross DB produced by cross_structure_builder.py.

Finds nearest experts in 3D Cross space by coordinate proximity,
or by named domain (maps domain string → approximate Cross coordinates).

Usage
-----
from knowledge.cross_searcher import CrossSearcher

searcher = CrossSearcher("cross_db.json")

# By coordinates
results = searcher.search(query_coords=(0.8, 0.3, 0.5), k=10)

# By domain name
results = searcher.search_by_features(domain="math", k=20)

for match in results:
    print(match)
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Optional fast nearest-neighbour index ─────────────────────────────────────
try:
    from scipy.spatial import KDTree
    KDTREE_AVAILABLE = True
except ImportError:
    KDTREE_AVAILABLE = False


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class ExpertMatch:
    """A single search result."""
    key:              str            # e.g. "L12E47"
    layer:            int
    expert_id:        int
    cross_coords:     Tuple[float, float, float]
    distance:         float
    features:         Dict[str, float] = field(default_factory=dict)
    # concept_directions are large; only included when explicitly requested
    concept_directions: Optional[List[List[float]]] = None

    def __repr__(self) -> str:
        x, y, z = self.cross_coords
        return (
            f"ExpertMatch(key={self.key!r}, "
            f"coords=({x:.3f}, {y:.3f}, {z:.3f}), "
            f"dist={self.distance:.4f})"
        )


# ── Domain → Cross coordinate map ─────────────────────────────────────────────
#
# Each domain maps to (X, Y, Z, radius) where radius is the search tolerance.
#
#   X – abstraction  (0=concrete/specialised, 1=abstract/broad)
#   Y – application  (0=diffuse/theoretical,  1=sharp/applied)
#   Z – depth        (0=shallow layers,        1=deep layers)
#
# Rationale
# ---------
# Math (pure)      : high abstraction, low application, all depths
# Math (applied)   : medium abstraction, high application
# Physics          : medium abstraction, high application, medium depth
# Chemistry        : low abstraction, high application, medium-shallow depth
# Biology          : low abstraction, high application, shallow
# Computer Science : medium abstraction, medium application, medium depth
# History/Language : low abstraction, high application, shallow-medium
# Reasoning/Logic  : high abstraction, low application, deep

DOMAIN_COORDS: Dict[str, Tuple[float, float, float, float]] = {
    # (X, Y, Z, search_radius)
    "math":              (0.80, 0.35, 0.50, 0.30),
    "mathematics":       (0.80, 0.35, 0.50, 0.30),
    "pure_math":         (0.85, 0.20, 0.55, 0.25),
    "applied_math":      (0.65, 0.70, 0.50, 0.25),
    "algebra":           (0.80, 0.30, 0.45, 0.25),
    "calculus":          (0.75, 0.40, 0.50, 0.25),
    "number_theory":     (0.90, 0.20, 0.60, 0.20),
    "linear_algebra":    (0.75, 0.45, 0.50, 0.25),
    "statistics":        (0.60, 0.65, 0.45, 0.25),
    "probability":       (0.65, 0.55, 0.45, 0.25),

    "physics":           (0.60, 0.75, 0.55, 0.30),
    "mechanics":         (0.55, 0.80, 0.45, 0.25),
    "quantum":           (0.75, 0.65, 0.65, 0.25),
    "thermodynamics":    (0.55, 0.75, 0.50, 0.25),
    "electrodynamics":   (0.60, 0.75, 0.55, 0.25),

    "chemistry":         (0.45, 0.80, 0.40, 0.25),
    "organic_chemistry": (0.40, 0.85, 0.35, 0.20),
    "biochemistry":      (0.45, 0.80, 0.40, 0.20),

    "biology":           (0.35, 0.80, 0.35, 0.25),
    "genetics":          (0.40, 0.75, 0.40, 0.20),
    "neuroscience":      (0.50, 0.70, 0.55, 0.25),

    "computer_science":  (0.65, 0.65, 0.50, 0.30),
    "cs":                (0.65, 0.65, 0.50, 0.30),
    "algorithms":        (0.70, 0.55, 0.55, 0.25),
    "programming":       (0.50, 0.75, 0.40, 0.25),
    "machine_learning":  (0.65, 0.70, 0.55, 0.25),

    "logic":             (0.90, 0.25, 0.65, 0.20),
    "reasoning":         (0.85, 0.30, 0.70, 0.25),
    "philosophy":        (0.85, 0.25, 0.60, 0.30),
    "ethics":            (0.75, 0.35, 0.55, 0.25),

    "history":           (0.30, 0.85, 0.30, 0.30),
    "language":          (0.40, 0.70, 0.35, 0.30),
    "linguistics":       (0.55, 0.55, 0.45, 0.25),
    "literature":        (0.35, 0.75, 0.35, 0.30),

    "coding":            (0.55, 0.75, 0.45, 0.25),
    "engineering":       (0.50, 0.80, 0.45, 0.25),
    "economics":         (0.55, 0.70, 0.45, 0.25),

    # Catch-all / generic
    "general":           (0.50, 0.50, 0.50, 0.50),
    "unknown":           (0.50, 0.50, 0.50, 0.50),
}


# ── CrossSearcher ──────────────────────────────────────────────────────────────

class CrossSearcher:
    """
    Nearest-expert search over the Cross DB.

    Parameters
    ----------
    cross_db_path : str
        Path to cross_db.json produced by cross_structure_builder.py.
    include_concept_directions : bool
        If True, ExpertMatch.concept_directions is populated (large).

    Notes
    -----
    On first load the class builds an in-memory index:
    - If scipy is available:  KDTree for O(log n) queries.
    - Otherwise:              brute-force O(n) scan.
    """

    def __init__(
        self,
        cross_db_path: str,
        include_concept_directions: bool = False,
    ) -> None:
        self._include_dirs = include_concept_directions
        self._db:     Dict[str, Any] = {}
        self._keys:   List[str]      = []
        self._coords: np.ndarray     = np.empty((0, 3))
        self._kdtree: Optional[Any]  = None

        self._load(cross_db_path)

    # ── Loading ────────────────────────────────────────────────────────────────

    def _load(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Cross DB not found: {path}")

        with open(path, "r") as f:
            raw = json.load(f)

        self._meta    = raw.get("metadata", {})
        self._experts = raw.get("experts", {})

        n = len(self._experts)
        if n == 0:
            raise ValueError("Cross DB contains no experts")

        # Build coordinate matrix
        self._keys   = list(self._experts.keys())
        self._coords = np.array(
            [self._experts[k].get("cross_coords", [0.5, 0.5, 0.5])
             for k in self._keys],
            dtype=np.float32,
        )

        # Build spatial index
        if KDTREE_AVAILABLE:
            self._kdtree = KDTree(self._coords)
            idx_type = "KDTree"
        else:
            idx_type = "brute-force"

        print(
            f"[CrossSearcher] Loaded {n} experts from {path}  "
            f"(index: {idx_type})"
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def search(
        self,
        query_coords: Tuple[float, float, float],
        k: int = 10,
        metric: str = "euclidean",
        layer_filter: Optional[int] = None,
    ) -> List[ExpertMatch]:
        """
        Find k nearest experts to query_coords in Cross space.

        Parameters
        ----------
        query_coords  : (X, Y, Z)  each in [0, 1]
        k             : number of results
        metric        : "euclidean" | "manhattan" | "cosine"
        layer_filter  : if set, only return experts from this layer

        Returns
        -------
        List[ExpertMatch] sorted by distance ascending.
        """
        q = np.array(query_coords, dtype=np.float32)
        q = np.clip(q, 0.0, 1.0)

        if metric == "euclidean" and self._kdtree is not None and layer_filter is None:
            distances, indices = self._kdtree.query(q, k=min(k, len(self._keys)))
            if isinstance(indices, (int, np.integer)):
                indices   = [indices]
                distances = [distances]
            pairs = list(zip(distances, indices))
        else:
            pairs = self._brute_force(q, k, metric, layer_filter)

        return [self._make_match(dist, idx) for dist, idx in pairs]

    def search_by_features(
        self,
        domain: str,
        k: int = 20,
        strict_radius: bool = False,
    ) -> List[ExpertMatch]:
        """
        Find k experts likely specialised in `domain`.

        Domain names are mapped to approximate Cross coordinates using a
        built-in table.  Unknown domains fall back to the centre (0.5, 0.5, 0.5).

        Parameters
        ----------
        domain        : domain name (e.g. "math", "physics", "coding")
        k             : number of results
        strict_radius : if True, filter to experts within the domain's radius

        Returns
        -------
        List[ExpertMatch] sorted by distance ascending.
        """
        key = domain.lower().strip()
        if key not in DOMAIN_COORDS:
            # Try prefix match
            for dk in DOMAIN_COORDS:
                if dk.startswith(key) or key.startswith(dk):
                    key = dk
                    break
            else:
                print(f"[CrossSearcher] Unknown domain '{domain}', using (0.5, 0.5, 0.5)")
                key = "general"

        x, y, z, radius = DOMAIN_COORDS[key]
        results = self.search(query_coords=(x, y, z), k=k)

        if strict_radius:
            results = [r for r in results if r.distance <= radius]

        return results

    def search_by_layer(
        self,
        layer: int,
        k: int = 256,
    ) -> List[ExpertMatch]:
        """Return all (or top-k) experts from a specific layer."""
        matched = [
            (0.0, i)
            for i, key in enumerate(self._keys)
            if self._experts[key].get("layer") == layer
        ][:k]
        return [self._make_match(dist, idx) for dist, idx in matched]

    def get_expert(self, key: str) -> Optional[Dict]:
        """Return raw expert dict by key (e.g. 'L12E47')."""
        return self._experts.get(key)

    def metadata(self) -> Dict:
        """Return Cross DB metadata."""
        return self._meta

    def stats(self) -> Dict:
        """Return summary statistics of the loaded Cross DB."""
        coords = self._coords
        return {
            "total_experts":  len(self._keys),
            "coord_mean":     coords.mean(axis=0).tolist(),
            "coord_std":      coords.std(axis=0).tolist(),
            "coord_min":      coords.min(axis=0).tolist(),
            "coord_max":      coords.max(axis=0).tolist(),
            "num_layers":     self._meta.get("num_layers"),
            "num_experts":    self._meta.get("num_experts"),
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    def _brute_force(
        self,
        q: np.ndarray,
        k: int,
        metric: str,
        layer_filter: Optional[int],
    ) -> List[Tuple[float, int]]:
        """O(n) distance computation."""
        if metric == "euclidean":
            diffs = self._coords - q
            dists = np.sqrt((diffs ** 2).sum(axis=1))
        elif metric == "manhattan":
            diffs = self._coords - q
            dists = np.abs(diffs).sum(axis=1)
        elif metric == "cosine":
            norms  = np.linalg.norm(self._coords, axis=1) + 1e-12
            q_norm = np.linalg.norm(q) + 1e-12
            dots   = (self._coords * q).sum(axis=1)
            dists  = 1.0 - (dots / (norms * q_norm))
        else:
            raise ValueError(f"Unknown metric: {metric!r}")

        if layer_filter is not None:
            mask = np.array(
                [self._experts[key].get("layer") == layer_filter
                 for key in self._keys],
                dtype=bool,
            )
            dists[~mask] = np.inf

        top_k_idx = np.argsort(dists)[:k]
        return [(float(dists[i]), int(i)) for i in top_k_idx]

    def _make_match(self, dist: float, idx: int) -> ExpertMatch:
        key    = self._keys[idx]
        expert = self._experts[key]
        coords = tuple(expert.get("cross_coords", [0.5, 0.5, 0.5]))
        concept_dirs = (
            expert.get("concept_directions")
            if self._include_dirs else None
        )
        return ExpertMatch(
            key               = key,
            layer             = expert.get("layer", -1),
            expert_id         = expert.get("expert_id", -1),
            cross_coords      = coords,  # type: ignore[arg-type]
            distance          = dist,
            features          = expert.get("features", {}),
            concept_directions= concept_dirs,
        )


# ── Convenience functions ──────────────────────────────────────────────────────

def load_searcher(cross_db_path: str = "cross_db.json") -> CrossSearcher:
    """Quick factory."""
    return CrossSearcher(cross_db_path)


def domain_coords(domain: str) -> Tuple[float, float, float]:
    """Return the canonical Cross coordinates for a domain name."""
    key = domain.lower().strip()
    if key not in DOMAIN_COORDS:
        return (0.5, 0.5, 0.5)
    x, y, z, _ = DOMAIN_COORDS[key]
    return (x, y, z)


# ── CLI demo ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Search Cross DB")
    parser.add_argument("--db",     default="cross_db.json", help="Path to cross_db.json")
    parser.add_argument("--domain", default=None,            help="Domain to search (e.g. math)")
    parser.add_argument("--coords", default=None,            help="Coords as 'X,Y,Z'  e.g. 0.8,0.3,0.5")
    parser.add_argument("--k",      default=10, type=int,    help="Number of results")
    parser.add_argument("--layer",  default=None, type=int,  help="Filter to specific layer")
    parser.add_argument("--stats",  action="store_true",     help="Print DB statistics")
    args = parser.parse_args()

    searcher = CrossSearcher(args.db)

    if args.stats:
        import pprint
        pprint.pprint(searcher.stats())
        raise SystemExit(0)

    if args.domain:
        print(f"\n[SEARCH] Domain: {args.domain!r}  (k={args.k})")
        x, y, z = domain_coords(args.domain)
        print(f"         → coords ({x:.3f}, {y:.3f}, {z:.3f})")
        results = searcher.search_by_features(args.domain, k=args.k)

    elif args.coords:
        vals  = [float(v) for v in args.coords.split(",")]
        query = (vals[0], vals[1], vals[2])
        print(f"\n[SEARCH] Coords: {query}  (k={args.k})")
        results = searcher.search(query, k=args.k, layer_filter=args.layer)

    else:
        print("Specify --domain or --coords.  Use --stats for DB overview.")
        raise SystemExit(1)

    print(f"\n{'Rank':<5} {'Key':<10} {'L':>4} {'E':>4}   "
          f"{'X':>6} {'Y':>6} {'Z':>6}   {'Dist':>8}")
    print("-" * 60)
    for rank, m in enumerate(results, 1):
        x, y, z = m.cross_coords
        print(f"{rank:<5} {m.key:<10} {m.layer:>4} {m.expert_id:>4}   "
              f"{x:>6.3f} {y:>6.3f} {z:>6.3f}   {m.distance:>8.4f}")
