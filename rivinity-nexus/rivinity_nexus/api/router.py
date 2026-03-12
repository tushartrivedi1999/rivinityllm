from fastapi import APIRouter

from rivinity_nexus.api.routes import auth, datasets, gpu, inference, jobs, models, training

api_router = APIRouter()
api_router.include_router(auth.router)
api_router.include_router(models.router)
api_router.include_router(datasets.router)
api_router.include_router(training.router)
api_router.include_router(inference.router)
api_router.include_router(gpu.router)
api_router.include_router(jobs.router)
