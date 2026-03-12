from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from rivinity_nexus.config.settings import get_settings

settings = get_settings()
engine = create_engine(settings.sqlalchemy_database_uri, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    from rivinity_nexus.models.entities import ApiKey, DatasetArtifact, GpuVendorNode, ModelArtifact, QueueJob, Session, TrainingJob, User  # noqa: F401

    Base.metadata.create_all(bind=engine)


def get_db_session() -> Generator:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
