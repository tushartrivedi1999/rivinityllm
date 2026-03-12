from pathlib import Path

from rivinity_nexus.config.settings import get_settings

try:
    import boto3
except Exception:  # pragma: no cover
    boto3 = None


class ModelStorageService:
    def __init__(self) -> None:
        self.settings = get_settings()

    def store_reference(self, owner_id: int, model_name: str, version: str, source_uri: str) -> str:
        key = f"models/{owner_id}/{model_name}/{version}/manifest.txt"
        if self.settings.s3_enabled and boto3:
            client = boto3.client(
                "s3",
                endpoint_url=self.settings.s3_endpoint_url,
                aws_access_key_id=self.settings.s3_access_key_id,
                aws_secret_access_key=self.settings.s3_secret_access_key,
                region_name=self.settings.s3_region,
            )
            body = source_uri.encode("utf-8")
            client.put_object(Bucket=self.settings.s3_bucket, Key=key, Body=body)
            return f"s3://{self.settings.s3_bucket}/{key}"

        root = Path(self.settings.model_storage_path) / str(owner_id) / model_name / version
        root.mkdir(parents=True, exist_ok=True)
        manifest = root / "manifest.txt"
        manifest.write_text(source_uri)
        return str(manifest)
