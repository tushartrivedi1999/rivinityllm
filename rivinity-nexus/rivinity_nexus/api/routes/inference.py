import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from rivinity_nexus.api.deps import get_current_user
from rivinity_nexus.data.schemas import InferenceRequest, InferenceResponse, MoERuntimeProfileResponse
from rivinity_nexus.engine.inference import GenerationRequest, InferenceBackend, InferenceEngine
from rivinity_nexus.models.entities import User
from rivinity_nexus.monitoring.metrics import record_tokens_per_second, timed

router = APIRouter(prefix="/inference", tags=["inference"])
engine = InferenceEngine()


@router.post("/generate", response_model=InferenceResponse)
def generate(payload: InferenceRequest, _: User = Depends(get_current_user)) -> InferenceResponse:
    prompts = payload.prompts or ([payload.prompt] if payload.prompt else [])
    if not prompts:
        raise HTTPException(status_code=422, detail="Either prompt or prompts is required")
    req = GenerationRequest(
        model_name=payload.model_name,
        prompts=prompts,
        max_new_tokens=payload.max_new_tokens,
        temperature=payload.temperature,
        backend=InferenceBackend(payload.backend),
        use_kv_cache=payload.use_kv_cache,
    )
    started = timed()
    result = engine.generate(req)
    elapsed = timed() - started
    record_tokens_per_second(
        backend=result.backend.value,
        model_name=payload.model_name,
        token_count=sum(result.token_counts),
        elapsed_seconds=elapsed,
    )
    return InferenceResponse(backend=result.backend.value, outputs=result.outputs, token_counts=result.token_counts)


@router.post("/stream")
def stream_generate(payload: InferenceRequest, _: User = Depends(get_current_user)) -> StreamingResponse:
    prompts = payload.prompts or ([payload.prompt] if payload.prompt else [])
    if not prompts:
        raise HTTPException(status_code=422, detail="Either prompt or prompts is required")
    req = GenerationRequest(
        model_name=payload.model_name,
        prompts=prompts,
        max_new_tokens=payload.max_new_tokens,
        temperature=payload.temperature,
        backend=InferenceBackend(payload.backend),
        use_kv_cache=payload.use_kv_cache,
    )

    def event_stream():
        for token in engine.stream_tokens(req):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/moe-profile", response_model=MoERuntimeProfileResponse)
def moe_profile(_: User = Depends(get_current_user)) -> MoERuntimeProfileResponse:
    return MoERuntimeProfileResponse(**engine.get_moe_runtime_profile())
