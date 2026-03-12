from celery import Celery

from rivinity_nexus.config.settings import get_settings

settings = get_settings()

celery_app = Celery(
    "rivinity_nexus",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_default_queue="rivinity_jobs",
)

celery_app.autodiscover_tasks(["rivinity_nexus.workers.tasks", "rivinity_nexus.workers.queue_tasks"])
