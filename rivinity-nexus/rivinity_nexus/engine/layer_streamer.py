from dataclasses import dataclass


@dataclass
class StreamStage:
    name: str


class LayerStreamer:
    """Dynamic expert/layer streaming pipeline: NVMe -> CPU -> GPU -> Evict."""

    def __init__(self) -> None:
        self.pipeline = [
            StreamStage("NVMe"),
            StreamStage("CPU cache"),
            StreamStage("GPU execution"),
            StreamStage("eviction"),
        ]

    def describe(self) -> str:
        return " -> ".join(stage.name for stage in self.pipeline)
