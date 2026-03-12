import asyncio
from collections import defaultdict


class ExpertRPCClient:
    """Asynchronous RPC simulation for distributed expert execution."""

    def __init__(self, batch_size: int = 16) -> None:
        self.batch_size = max(1, batch_size)

    def _chunk(self, values: list[str]) -> list[list[str]]:
        return [values[i : i + self.batch_size] for i in range(0, len(values), self.batch_size)]

    async def send_token_batch(self, node_id: str, token_batch: list[str], expert_id: int) -> dict:
        await asyncio.sleep(0)
        return {
            "node_id": node_id,
            "expert_id": expert_id,
            "outputs": [f"{token}|expert={expert_id}|node={node_id}" for token in token_batch],
        }

    async def gather(self, requests: list[tuple[str, list[str], int]]) -> list[dict]:
        tasks = [self.send_token_batch(node, batch, expert) for node, batch, expert in requests]
        return await asyncio.gather(*tasks)

    async def execute_distributed(self, assignments: dict[int, str], token_stream: list[str]) -> dict[int, list[str]]:
        """Route token batches to remote expert nodes and collect outputs."""
        batched_requests: list[tuple[str, list[str], int]] = []
        for expert_id, node_id in assignments.items():
            for token_batch in self._chunk(token_stream):
                batched_requests.append((node_id, token_batch, expert_id))

        results = await self.gather(batched_requests)
        merged: dict[int, list[str]] = defaultdict(list)
        for result in results:
            merged[result["expert_id"]].extend(result["outputs"])
        return dict(merged)
