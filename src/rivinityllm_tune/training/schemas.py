"""Schemas for RL post-training jobs."""

from typing import Literal

from pydantic import BaseModel, Field


class RLTrainRequest(BaseModel):
    base_model: str = Field(description="Base model id or path")
    environment: str = Field(description="RL environment identifier")
    algorithm: Literal["ppo", "grpo", "dpo"] = "ppo"
    max_steps: int = Field(default=1000, ge=1)


class RLTrainResponse(BaseModel):
    job_id: str
    status: Literal["queued"] = "queued"
