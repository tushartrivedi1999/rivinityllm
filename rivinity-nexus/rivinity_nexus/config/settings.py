from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    app_name: str = "Rivinity Nexus"
    api_v1_prefix: str = ""
    log_level: str = "INFO"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "rivinity_nexus"
    postgres_user: str = "rivinity"
    postgres_password: str = "rivinity_password"
    database_url: str | None = None

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    jwt_secret_key: str = "change_me"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60
    jwt_refresh_token_expire_days: int = 7
    api_key_header_name: str = "X-API-Key"

    gpu_vendor_api_base: str = "https://vendor.example.com/api"
    gpu_vendor_api_token: str = "token"
    model_storage_path: str = "/tmp/rivinity/models"
    dataset_storage_path: str = "/tmp/rivinity/datasets"
    dataset_cache_path: str = "/tmp/rivinity/cache/datasets"
    dataset_stream_batch_size: int = 1000
    s3_enabled: bool = False
    s3_endpoint_url: str | None = None
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    s3_bucket: str = "rivinity-models"
    s3_region: str = "us-east-1"

    experiment_tracking_enabled: bool = False
    experiment_tracking_backend: str = "mlflow"
    mlflow_tracking_uri: str = "file:/tmp/rivinity/mlruns"
    mlflow_experiment_name: str = "rivinity-training"

    @property
    def sqlalchemy_database_uri(self) -> str:
        if self.database_url:
            return self.database_url
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
