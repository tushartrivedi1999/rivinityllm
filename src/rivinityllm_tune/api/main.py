"""FastAPI entrypoint for RivinityLLM Tune."""

from __future__ import annotations

from hashlib import sha256

from fastapi import FastAPI, HTTPException

from rivinityllm_tune.auth.models import TokenResponse, UserCreate, UserLogin, UserPublic
from rivinityllm_tune.auth.security import create_access_token
from rivinityllm_tune.auth.store import user_store
from rivinityllm_tune.inference.planner import PlanError, build_inference_plan
from rivinityllm_tune.inference.schemas import (
    InferencePlanRequest,
    InferencePlanResponse,
    InferenceRequest,
    InferenceResponse,
)
from rivinityllm_tune.training.schemas import RLTrainRequest, RLTrainResponse


def _stable_job_id(seed: str) -> str:
    digest = sha256(seed.encode("utf-8")).hexdigest()[:12]
    return f"rl-{digest}"


def create_app() -> FastAPI:
    app = FastAPI(title="RivinityLLM Tune", version="0.3.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/auth/signup", response_model=UserPublic)
    def signup(payload: UserCreate) -> UserPublic:
        try:
            user = user_store.create_user(payload.email, payload.password)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return UserPublic(email=user.email)

    @app.post("/auth/login", response_model=TokenResponse)
    def login(payload: UserLogin) -> TokenResponse:
        if not user_store.authenticate(payload.email, payload.password):
            raise HTTPException(status_code=401, detail="Invalid credentials")
        token = create_access_token(payload.email)
        return TokenResponse(access_token=token)

    @app.post("/v1/train/rl", response_model=RLTrainResponse)
    def enqueue_rl_training(payload: RLTrainRequest) -> RLTrainResponse:
        job_seed = f"{payload.base_model}:{payload.environment}:{payload.algorithm}:{payload.max_steps}"
        return RLTrainResponse(job_id=_stable_job_id(job_seed))

    @app.post("/v1/inference/generate", response_model=InferenceResponse)
    def generate(payload: InferenceRequest) -> InferenceResponse:
        strategy = "cpu_offload+paged_weights" if payload.memory_budget_gb < 12 else "gpu_prefetch"
        return InferenceResponse(
            output=f"[demo-output for {payload.model_id}] {payload.prompt}",
            strategy=strategy,
        )

    @app.post("/v1/inference/plan", response_model=InferencePlanResponse)
    def plan_inference(payload: InferencePlanRequest) -> InferencePlanResponse:
        try:
            plan = build_inference_plan(
                parameter_count_b=payload.parameter_count_b,
                quantization_bits=payload.quantization_bits,
                target_gpu_vram_gb=payload.target_gpu_vram_gb,
                sequence_length=payload.sequence_length,
                batch_size=payload.batch_size,
            )
        except PlanError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return InferencePlanResponse(**plan.__dict__)

    return app


app = create_app()
