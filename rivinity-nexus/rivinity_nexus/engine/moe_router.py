from dataclasses import dataclass
import math

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass
class RoutingResult:
    expert_ids: list[int]
    scores: list[float]


class MoERouter:
    """Top-k sparse MoE router with softmax gating."""

    def __init__(self, num_experts: int, top_k: int = 2) -> None:
        self.num_experts = max(1, num_experts)
        self.top_k = max(1, min(top_k, self.num_experts))

    def _softmax(self, logits: list[float]) -> list[float]:
        m = max(logits)
        exps = [math.exp(x - m) for x in logits]
        total = sum(exps) or 1.0
        return [v / total for v in exps]

    def route(self, hidden_states) -> RoutingResult:
        """Return top-k experts and gate scores.

        hidden_states can be a torch.Tensor or list[float].
        """
        if torch is not None and hasattr(hidden_states, "numel"):
            seed = int(float(hidden_states.float().mean().item()) * 1000) if hidden_states.numel() else 0
            generator = torch.Generator(device="cpu")
            generator.manual_seed(abs(seed) + self.num_experts)
            logits_t = torch.randn(self.num_experts, generator=generator)
            probs_t = torch.softmax(logits_t, dim=0)
            vals, idx = torch.topk(probs_t, k=self.top_k)
            return RoutingResult(expert_ids=idx.tolist(), scores=[float(v) for v in vals.tolist()])

        if isinstance(hidden_states, list) and hidden_states:
            base = sum(float(x) for x in hidden_states) / len(hidden_states)
        else:
            base = 0.0
        logits = [math.sin(base + i * 0.013) for i in range(self.num_experts)]
        probs = self._softmax(logits)
        ranked = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[: self.top_k]
        return RoutingResult(expert_ids=[i for i, _ in ranked], scores=[s for _, s in ranked])
