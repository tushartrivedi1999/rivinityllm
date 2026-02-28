"""Application settings."""

from pydantic import BaseModel, Field


class Settings(BaseModel):
    app_name: str = Field(default="RivinityLLM Tune")
    jwt_secret: str = Field(default="change-me-in-production")
    jwt_algorithm: str = Field(default="HS256")
    access_token_ttl_minutes: int = Field(default=60)


settings = Settings()
