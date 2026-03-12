class ExpertScheduler:
    """Assign routed experts to available GPU nodes."""

    def assign(self, expert_ids: list[int], available_nodes: list[str]) -> dict[int, str]:
        if not available_nodes:
            return {eid: "unassigned" for eid in expert_ids}
        mapping: dict[int, str] = {}
        for idx, eid in enumerate(expert_ids):
            mapping[eid] = available_nodes[idx % len(available_nodes)]
        return mapping
