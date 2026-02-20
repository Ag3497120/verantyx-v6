"""
Expert Router Analyzer

The RIGHT way to find domain experts: use ACTUAL ROUTING ANALYSIS.
Instead of fake weight signatures, we run probe queries through DeepSeek V3
and record which experts fire most often for each domain.

Usage:
    analyzer = ExpertRouterAnalyzer(model=None, stub=True)
    routing_map = analyzer.analyze_all_domains()
    top_experts = analyzer.get_top_experts("math", n=20)
"""

import json
import random
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path

# ──────────────────────────────────────────────
# Domain Probe Queries
# ──────────────────────────────────────────────

DOMAIN_PROBES: Dict[str, List[str]] = {
    "math": [
        "Solve: x^2 + 5x + 6 = 0",
        "What is the derivative of sin(x)?",
        "Find the determinant of [[1,2],[3,4]]",
        "Integrate: ∫ x^2 dx from 0 to 3",
        "Prove that √2 is irrational",
        "What is the sum of the first 100 natural numbers?",
        "Solve the system: 2x + y = 5, x - y = 1",
        "Find all prime numbers less than 30",
        "Compute gcd(48, 18)",
        "What is the eigenvalue of [[2,1],[1,2]]?",
        "Expand (a+b)^4 using the binomial theorem",
        "Solve: log_2(x) = 5",
        "Find the limit: lim_{x→0} sin(x)/x",
        "Compute the cross product of [1,0,0] and [0,1,0]",
        "What is Euler's formula e^(iπ) + 1 = 0?",
    ],
    "physics": [
        "F = ma, find a if F=10N, m=2kg",
        "What is the speed of light in vacuum?",
        "A ball is dropped from 20m. How long until it hits the ground?",
        "Calculate the kinetic energy of a 5kg object moving at 10m/s",
        "What is Ohm's Law?",
        "Explain the photoelectric effect",
        "What is the wavelength of a photon with energy 3eV?",
        "Calculate the escape velocity from Earth",
        "What is Heisenberg's uncertainty principle?",
        "Find the period of a pendulum of length 1m",
    ],
    "chemistry": [
        "Balance: H2 + O2 → H2O",
        "What is the pH of 0.01 mol/L HCl?",
        "What is the electron configuration of carbon?",
        "How many moles in 18g of water?",
        "What is the oxidation state of S in H2SO4?",
        "Write the molecular formula of glucose",
        "What is Le Chatelier's principle?",
        "Calculate the molar mass of CaCO3",
        "Balance: Fe + HCl → FeCl2 + H2",
        "What type of bond is found in NaCl?",
    ],
    "biology": [
        "What is the central dogma of molecular biology?",
        "How many chromosomes does a human cell have?",
        "What is the function of mitochondria?",
        "Explain DNA replication",
        "What is natural selection?",
        "What is the role of ATP in cells?",
        "What is the difference between meiosis and mitosis?",
        "What is the Hardy-Weinberg principle?",
        "Describe the structure of a neuron",
        "What is homeostasis?",
    ],
    "computer_science": [
        "What is the time complexity of merge sort?",
        "Implement binary search in pseudocode",
        "What is a hash table?",
        "Explain the difference between TCP and UDP",
        "What is a Turing machine?",
        "What is Big-O notation?",
        "Explain depth-first search",
        "What is a binary tree?",
        "What is dynamic programming?",
        "Explain the P vs NP problem",
    ],
    "history": [
        "When did World War II end?",
        "Who was the first US President?",
        "What caused the French Revolution?",
        "When did the Roman Empire fall?",
        "Who invented the printing press?",
        "What was the Cold War?",
        "When did India gain independence?",
        "Who was Napoleon Bonaparte?",
        "What was the Industrial Revolution?",
        "When was the Magna Carta signed?",
    ],
    "literature": [
        "Who wrote Hamlet?",
        "What is the theme of 1984 by George Orwell?",
        "Identify the literary device in 'The fog comes on little cat feet'",
        "What is the protagonist of Crime and Punishment?",
        "Explain the symbolism of the green light in The Great Gatsby",
        "Who wrote The Iliad?",
        "What is a haiku?",
        "Describe the plot of Don Quixote",
        "What is magical realism?",
        "Who wrote Pride and Prejudice?",
    ],
    "philosophy": [
        "What is Descartes' cogito ergo sum?",
        "Explain Plato's allegory of the cave",
        "What is utilitarianism?",
        "What is the trolley problem?",
        "Explain Kant's categorical imperative",
        "What is existentialism?",
        "What is the problem of induction?",
        "Explain the ship of Theseus paradox",
        "What is Hume's fork?",
        "What is the mind-body problem?",
    ],
}


# ──────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────

@dataclass
class ExpertActivation:
    """Records a single expert activation during a forward pass"""
    layer: int
    expert_id: int
    activation_score: float
    probe_query: str
    domain: str


@dataclass
class DomainExpertProfile:
    """Profile of an expert for a given domain"""
    layer: int
    expert_id: int
    domain: str
    activation_frequency: float   # fraction of domain probes that activated this expert
    mean_activation_score: float  # average activation strength
    query_count: int              # number of probe queries processed


@dataclass
class RoutingAnalysisResult:
    """Full result of routing analysis across all domains"""
    domain_experts: Dict[str, List[DomainExpertProfile]]
    raw_activations: List[ExpertActivation]
    num_layers: int
    num_experts_per_layer: int
    probe_counts: Dict[str, int]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────
# Expert Router Analyzer
# ──────────────────────────────────────────────

class ExpertRouterAnalyzer:
    """
    Analyzes expert routing in DeepSeek V3 MoE model.

    In STUB mode: generates realistic synthetic routing data deterministically
    based on probe query content (domain-specific hash seeds).

    In REAL mode: requires a loaded DeepSeek V3 model with hooks to capture
    expert selection in each MoE layer.
    """

    # DeepSeek V3 architecture
    NUM_LAYERS = 61
    NUM_EXPERTS = 256
    TOP_K_EXPERTS = 8   # DeepSeek V3 activates top-8 experts per token

    def __init__(
        self,
        model=None,
        tokenizer=None,
        stub: bool = False,
        cache_path: Optional[str] = None,
    ):
        """
        Args:
            model: Loaded DeepSeek V3 model (None in stub mode)
            tokenizer: Corresponding tokenizer
            stub: If True, run in stub mode (no real model needed)
            cache_path: Path to cache routing results
        """
        self.model = model
        self.tokenizer = tokenizer
        self.stub = stub
        self.cache_path = cache_path

        self._routing_hooks = []
        self._activation_buffer: List[ExpertActivation] = []

        mode = "STUB" if stub else "REAL"
        print(f"[EXPERT_ROUTER] Initialized in {mode} mode")
        print(f"[EXPERT_ROUTER] Architecture: {self.NUM_LAYERS} layers × {self.NUM_EXPERTS} experts")

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def analyze_all_domains(
        self,
        domains: Optional[List[str]] = None,
        probes_per_domain: int = 15,
    ) -> RoutingAnalysisResult:
        """
        Run routing analysis across all (or specified) domains.

        Args:
            domains: List of domains to analyze. None = all domains.
            probes_per_domain: Number of probe queries per domain.

        Returns:
            RoutingAnalysisResult with per-domain expert profiles.
        """
        domains = domains or list(DOMAIN_PROBES.keys())
        all_activations: List[ExpertActivation] = []
        probe_counts: Dict[str, int] = {}

        print(f"[EXPERT_ROUTER] Analyzing {len(domains)} domains...")

        for domain in domains:
            probes = DOMAIN_PROBES.get(domain, [])[:probes_per_domain]
            probe_counts[domain] = len(probes)

            print(f"[EXPERT_ROUTER]   {domain}: {len(probes)} probes")
            activations = self._run_probes(domain, probes)
            all_activations.extend(activations)

        # Aggregate into per-domain expert profiles
        domain_experts = self._aggregate_activations(all_activations, domains)

        result = RoutingAnalysisResult(
            domain_experts=domain_experts,
            raw_activations=all_activations,
            num_layers=self.NUM_LAYERS,
            num_experts_per_layer=self.NUM_EXPERTS,
            probe_counts=probe_counts,
            metadata={
                "stub_mode": self.stub,
                "top_k": self.TOP_K_EXPERTS,
            }
        )

        if self.cache_path:
            self._save_result(result, self.cache_path)

        return result

    def get_top_experts(
        self,
        domain: str,
        result: Optional[RoutingAnalysisResult] = None,
        n: int = 20,
    ) -> List[DomainExpertProfile]:
        """
        Get top N experts for a domain, sorted by activation frequency.

        Args:
            domain: Domain name (e.g. "math", "physics")
            result: Pre-computed RoutingAnalysisResult. If None, runs analysis.
            n: Number of experts to return.

        Returns:
            Sorted list of DomainExpertProfile (highest frequency first).
        """
        if result is None:
            result = self.analyze_all_domains(domains=[domain])

        experts = result.domain_experts.get(domain, [])
        return sorted(experts, key=lambda e: -e.activation_frequency)[:n]

    # ──────────────────────────────────────────
    # Probe Execution
    # ──────────────────────────────────────────

    def _run_probes(
        self,
        domain: str,
        probes: List[str],
    ) -> List[ExpertActivation]:
        """Run probe queries and collect expert activations."""
        if self.stub:
            return self._stub_run_probes(domain, probes)
        else:
            return self._real_run_probes(domain, probes)

    def _stub_run_probes(
        self,
        domain: str,
        probes: List[str],
    ) -> List[ExpertActivation]:
        """
        Stub mode: generate realistic synthetic routing data.

        Uses deterministic hash-based seeding so results are reproducible
        and domain-specific (math probes tend to activate different experts
        than history probes).
        """
        activations = []

        # Domain-specific expert clusters (simulate specialization)
        domain_seed = int(hashlib.md5(domain.encode()).hexdigest()[:8], 16)
        rng = random.Random(domain_seed)

        # Each domain has "preferred" expert clusters
        # Experts at specific layers are more likely to fire
        domain_preferred_layers = sorted(rng.sample(range(self.NUM_LAYERS), 15))
        domain_preferred_experts = sorted(rng.sample(range(self.NUM_EXPERTS), 30))

        for probe in probes:
            # Per-probe variation
            probe_seed = int(hashlib.md5(probe.encode()).hexdigest()[:8], 16)
            probe_rng = random.Random(probe_seed ^ domain_seed)

            # Simulate top-K expert routing per token (average 20 tokens per probe)
            num_tokens = probe_rng.randint(10, 35)

            for token_idx in range(num_tokens):
                for layer in range(self.NUM_LAYERS):
                    # Skip non-MoE layers (first few are always dense)
                    if layer < 3:
                        continue

                    # Determine which experts activate for this token/layer
                    # Domain experts fire with higher probability
                    selected_experts = []

                    for _ in range(self.TOP_K_EXPERTS):
                        if probe_rng.random() < 0.4:
                            # Pick from domain-preferred experts
                            exp_id = probe_rng.choice(domain_preferred_experts)
                        else:
                            exp_id = probe_rng.randint(0, self.NUM_EXPERTS - 1)

                        if layer in domain_preferred_layers and probe_rng.random() < 0.3:
                            # Boost: pick from domain-preferred experts at preferred layers
                            exp_id = probe_rng.choice(domain_preferred_experts)

                        if exp_id not in selected_experts:
                            selected_experts.append(exp_id)

                    for exp_id in selected_experts:
                        activation_score = probe_rng.gauss(0.6, 0.15)
                        activation_score = max(0.1, min(1.0, activation_score))

                        activations.append(ExpertActivation(
                            layer=layer,
                            expert_id=exp_id,
                            activation_score=activation_score,
                            probe_query=probe[:50],
                            domain=domain,
                        ))

        return activations

    def _real_run_probes(
        self,
        domain: str,
        probes: List[str],
    ) -> List[ExpertActivation]:
        """
        Real mode: run actual forward passes through DeepSeek V3.

        Requires model to be loaded and hooks registered.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Use stub=True for testing.")

        activations = []
        self._activation_buffer = []
        self._register_routing_hooks()

        try:
            import torch
            for probe in probes:
                inputs = self.tokenizer(probe, return_tensors="pt")
                with torch.no_grad():
                    _ = self.model(**inputs)

                # Collect from buffer
                for act in self._activation_buffer:
                    act.probe_query = probe[:50]
                    act.domain = domain
                    activations.append(act)

                self._activation_buffer = []
        finally:
            self._remove_routing_hooks()

        return activations

    def _register_routing_hooks(self):
        """Register forward hooks to capture expert routing decisions."""
        # Hook into MoE router layers
        # DeepSeek V3 uses: model.layers[i].mlp.gate for routing
        for layer_idx, layer_module in enumerate(self.model.model.layers):
            if hasattr(layer_module, 'mlp') and hasattr(layer_module.mlp, 'gate'):
                hook = layer_module.mlp.gate.register_forward_hook(
                    self._make_routing_hook(layer_idx)
                )
                self._routing_hooks.append(hook)

    def _make_routing_hook(self, layer_idx: int):
        """Create a forward hook for a specific MoE layer."""
        def hook_fn(module, input, output):
            # output shape: [batch, seq, num_experts]
            import torch
            if isinstance(output, tuple):
                routing_weights = output[0]
            else:
                routing_weights = output

            # Get top-k expert indices
            top_k_weights, top_k_indices = torch.topk(
                routing_weights, self.TOP_K_EXPERTS, dim=-1
            )

            # Flatten batch/seq dims, record each activation
            flat_indices = top_k_indices.reshape(-1, self.TOP_K_EXPERTS)
            flat_weights = top_k_weights.reshape(-1, self.TOP_K_EXPERTS)

            for token_experts, token_weights in zip(flat_indices, flat_weights):
                for exp_id, weight in zip(token_experts.tolist(), token_weights.tolist()):
                    self._activation_buffer.append(ExpertActivation(
                        layer=layer_idx,
                        expert_id=int(exp_id),
                        activation_score=float(weight),
                        probe_query="",  # filled in caller
                        domain="",       # filled in caller
                    ))

        return hook_fn

    def _remove_routing_hooks(self):
        """Remove all registered hooks."""
        for hook in self._routing_hooks:
            hook.remove()
        self._routing_hooks = []

    # ──────────────────────────────────────────
    # Aggregation
    # ──────────────────────────────────────────

    def _aggregate_activations(
        self,
        activations: List[ExpertActivation],
        domains: List[str],
    ) -> Dict[str, List[DomainExpertProfile]]:
        """
        Aggregate raw activations into per-domain expert profiles.

        Returns {domain: [DomainExpertProfile sorted by frequency desc]}
        """
        # Count activations: {domain: {(layer, exp_id): [scores]}}
        counts: Dict[str, Dict[Tuple[int, int], List[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        domain_probe_counts: Dict[str, int] = defaultdict(set)

        for act in activations:
            counts[act.domain][(act.layer, act.expert_id)].append(act.activation_score)
            domain_probe_counts[act.domain].add(act.probe_query)

        result: Dict[str, List[DomainExpertProfile]] = {}

        for domain in domains:
            domain_counts = counts.get(domain, {})
            num_probes = max(1, len(domain_probe_counts.get(domain, set())))
            profiles = []

            for (layer, exp_id), scores in domain_counts.items():
                # Approximate frequency as fraction of probes that activated this expert
                # (rough approximation: num_activations / (num_probes * avg_tokens * layers))
                freq = len(scores) / (num_probes * 20 * self.TOP_K_EXPERTS + 1)
                freq = min(1.0, freq)

                profiles.append(DomainExpertProfile(
                    layer=layer,
                    expert_id=exp_id,
                    domain=domain,
                    activation_frequency=freq,
                    mean_activation_score=sum(scores) / len(scores),
                    query_count=num_probes,
                ))

            # Sort by frequency descending
            profiles.sort(key=lambda p: -p.activation_frequency)
            result[domain] = profiles

        return result

    # ──────────────────────────────────────────
    # Serialization
    # ──────────────────────────────────────────

    def _save_result(self, result: RoutingAnalysisResult, path: str):
        """Save routing analysis result to JSON."""
        data = {
            "metadata": result.metadata,
            "num_layers": result.num_layers,
            "num_experts_per_layer": result.num_experts_per_layer,
            "probe_counts": result.probe_counts,
            "domain_experts": {
                domain: [
                    {
                        "layer": p.layer,
                        "expert_id": p.expert_id,
                        "domain": p.domain,
                        "activation_frequency": p.activation_frequency,
                        "mean_activation_score": p.mean_activation_score,
                        "query_count": p.query_count,
                    }
                    for p in profiles[:50]  # save top-50 per domain
                ]
                for domain, profiles in result.domain_experts.items()
            }
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[EXPERT_ROUTER] Saved routing analysis to {path}")

    @staticmethod
    def load_result(path: str) -> RoutingAnalysisResult:
        """Load a previously saved routing analysis result."""
        with open(path) as f:
            data = json.load(f)

        domain_experts = {}
        for domain, profiles_data in data.get("domain_experts", {}).items():
            domain_experts[domain] = [
                DomainExpertProfile(**p) for p in profiles_data
            ]

        return RoutingAnalysisResult(
            domain_experts=domain_experts,
            raw_activations=[],
            num_layers=data.get("num_layers", 61),
            num_experts_per_layer=data.get("num_experts_per_layer", 256),
            probe_counts=data.get("probe_counts", {}),
            metadata=data.get("metadata", {}),
        )


# ──────────────────────────────────────────────
# CLI / Quick Test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Expert Router Analyzer - Stub Test ===")
    analyzer = ExpertRouterAnalyzer(stub=True)

    result = analyzer.analyze_all_domains(probes_per_domain=5)

    for domain in ["math", "physics", "chemistry"]:
        top = analyzer.get_top_experts(domain, result, n=5)
        print(f"\nTop experts for [{domain}]:")
        for p in top:
            print(f"  L{p.layer:02d}E{p.expert_id:03d}  freq={p.activation_frequency:.4f}  score={p.mean_activation_score:.3f}")
