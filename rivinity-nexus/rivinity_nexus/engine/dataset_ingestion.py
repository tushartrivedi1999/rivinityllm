import hashlib
import json
from pathlib import Path
from typing import Iterator

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from rivinity_nexus.config.settings import get_settings
from rivinity_nexus.models.entities import DatasetArtifact, DatasetFormat, DatasetStatus

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover
    load_dataset = None


class DatasetIngestionService:
    def __init__(self, db: Session):
        self.db = db
        self.settings = get_settings()

    def _cache_key(self, source_uri: str, fmt: DatasetFormat) -> str:
        return hashlib.sha256(f"{fmt.value}:{source_uri}".encode("utf-8")).hexdigest()

    def _storage_uri(self, owner_id: int, dataset_name: str, version: str) -> str:
        base = Path(self.settings.dataset_storage_path) / str(owner_id) / dataset_name / version
        base.mkdir(parents=True, exist_ok=True)
        return str(base)

    def upload_dataset(
        self,
        owner_id: int,
        dataset_name: str,
        version: str,
        dataset_format: DatasetFormat,
        source_uri: str,
    ) -> DatasetArtifact:
        existing = (
            self.db.query(DatasetArtifact)
            .filter(
                DatasetArtifact.owner_id == owner_id,
                DatasetArtifact.dataset_name == dataset_name,
                DatasetArtifact.version == version,
            )
            .first()
        )
        if existing:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Dataset version already exists")

        artifact = DatasetArtifact(
            dataset_name=dataset_name,
            version=version,
            format=dataset_format,
            source_uri=source_uri,
            storage_uri=self._storage_uri(owner_id, dataset_name, version),
            cache_key=self._cache_key(source_uri, dataset_format),
            status=DatasetStatus.uploaded,
            owner_id=owner_id,
        )
        self.db.add(artifact)
        self.db.commit()
        self.db.refresh(artifact)
        return artifact

    def list_datasets(self, owner_id: int) -> list[DatasetArtifact]:
        return (
            self.db.query(DatasetArtifact)
            .filter(DatasetArtifact.owner_id == owner_id)
            .order_by(DatasetArtifact.dataset_name.asc(), DatasetArtifact.upload_date.desc())
            .all()
        )

    def _stream_jsonl(self, source_uri: str) -> Iterator[dict]:
        with open(source_uri, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def _stream_parquet(self, source_uri: str) -> Iterator[dict]:
        if pq is None:
            raise HTTPException(status_code=500, detail="Parquet support requires pyarrow")
        parquet_file = pq.ParquetFile(source_uri)
        for batch in parquet_file.iter_batches(batch_size=self.settings.dataset_stream_batch_size):
            for row in batch.to_pylist():
                yield row

    def _stream_hf(self, source_uri: str) -> Iterator[dict]:
        if load_dataset is None:
            raise HTTPException(status_code=500, detail="HuggingFace dataset support requires datasets package")
        ds = load_dataset(source_uri, split="train", streaming=True)
        for item in ds:
            yield dict(item)

    def stream_dataset(self, artifact: DatasetArtifact) -> Iterator[dict]:
        if artifact.format == DatasetFormat.jsonl:
            return self._stream_jsonl(artifact.source_uri)
        if artifact.format == DatasetFormat.parquet:
            return self._stream_parquet(artifact.source_uri)
        return self._stream_hf(artifact.source_uri)

    def preprocess_and_shard(self, artifact: DatasetArtifact, shard_count: int) -> DatasetArtifact:
        shard_dir = Path(artifact.storage_uri) / "preprocessed"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shards = [open(shard_dir / f"shard_{idx}.jsonl", "w", encoding="utf-8") for idx in range(shard_count)]
        try:
            for idx, row in enumerate(self.stream_dataset(artifact)):
                processed = {k: v for k, v in row.items() if v is not None}
                shard_id = idx % shard_count
                shards[shard_id].write(json.dumps(processed) + "\n")
        finally:
            for fp in shards:
                fp.close()

        artifact.shard_count = shard_count
        artifact.preprocessed_uri = str(shard_dir)
        artifact.status = DatasetStatus.preprocessed
        self.db.commit()
        self.db.refresh(artifact)
        return artifact

    def get_distributed_shard_path(self, artifact: DatasetArtifact, rank: int, world_size: int) -> str:
        if not artifact.preprocessed_uri:
            raise HTTPException(status_code=400, detail="Dataset is not preprocessed")
        shard_id = rank % max(1, min(world_size, artifact.shard_count))
        return str(Path(artifact.preprocessed_uri) / f"shard_{shard_id}.jsonl")
