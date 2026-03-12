import json

import redis

from rivinity_nexus.config.settings import get_settings

settings = get_settings()


def get_redis_client() -> redis.Redis:
    return redis.Redis(host=settings.redis_host, port=settings.redis_port, db=settings.redis_db, decode_responses=True)


def publish_job_progress(job_id: int, state: str, progress: int, message: str | None = None) -> None:
    payload = {"job_id": job_id, "state": state, "progress": progress, "message": message}
    get_redis_client().publish("rivinity:jobs:progress", json.dumps(payload))
