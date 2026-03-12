from dataclasses import dataclass
from enum import Enum
import os


class LauncherType(str, Enum):
    torchrun = "torchrun"
    deepspeed = "deepspeed"
    ray = "ray"


@dataclass
class DistributedConfig:
    launcher: LauncherType = LauncherType.torchrun
    nnodes: int = 1
    nproc_per_node: int = 1
    node_rank: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    backend: str = "nccl"

    @property
    def world_size(self) -> int:
        return self.nnodes * self.nproc_per_node


class DistributedTrainingLauncher:
    """Build launcher commands and env for distributed training backends."""

    def _auto_from_env(self, cfg: DistributedConfig) -> DistributedConfig:
        cfg.node_rank = int(os.getenv("RANK", cfg.node_rank))
        cfg.master_addr = os.getenv("MASTER_ADDR", cfg.master_addr)
        cfg.master_port = int(os.getenv("MASTER_PORT", str(cfg.master_port)))
        env_world_size = os.getenv("WORLD_SIZE")
        if env_world_size:
            world_size = int(env_world_size)
            cfg.nnodes = max(1, world_size // max(1, cfg.nproc_per_node))
        return cfg

    def resolve_env(self, cfg: DistributedConfig) -> dict[str, str]:
        cfg = self._auto_from_env(cfg)
        return {
            "RANK": str(cfg.node_rank),
            "WORLD_SIZE": str(cfg.world_size),
            "MASTER_ADDR": cfg.master_addr,
            "MASTER_PORT": str(cfg.master_port),
            "TORCH_DISTRIBUTED_BACKEND": cfg.backend,
            "NCCL_DEBUG": "WARN",
        }

    def build_launch_command(self, cfg: DistributedConfig, training_entrypoint: str, entrypoint_args: list[str]) -> list[str]:
        cfg = self._auto_from_env(cfg)

        if cfg.launcher == LauncherType.torchrun:
            return [
                "torchrun",
                f"--nnodes={cfg.nnodes}",
                f"--nproc-per-node={cfg.nproc_per_node}",
                f"--node-rank={cfg.node_rank}",
                f"--master-addr={cfg.master_addr}",
                f"--master-port={cfg.master_port}",
                training_entrypoint,
                *entrypoint_args,
            ]

        if cfg.launcher == LauncherType.deepspeed:
            return [
                "deepspeed",
                f"--num_nodes={cfg.nnodes}",
                f"--num_gpus={cfg.nproc_per_node}",
                "--master_addr",
                cfg.master_addr,
                "--master_port",
                str(cfg.master_port),
                training_entrypoint,
                *entrypoint_args,
            ]

        return [
            "ray",
            "job",
            "submit",
            "--address",
            os.getenv("RAY_ADDRESS", "auto"),
            "--",
            "python",
            training_entrypoint,
            *entrypoint_args,
        ]
