import boto3
from config import Config
from urllib.parse import urlparse
from pathlib import Path


def parse_s3_uri(s3_uri: str):
    """Split s3://bucket/prefix into (bucket, prefix)."""
    parsed = urlparse(s3_uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    bucket = parsed.netloc
    prefix = parsed.path.lstrip("/")
    return bucket, prefix

def download_s3_folder(s3_client, bucket: str, prefix: str, local_dir: str):
    """Recursively download an entire S3 'folder'."""
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # skip directory placeholders
            if key.endswith("/"):
                continue

            # Build local path
            rel_path = key[len(prefix):].lstrip("/")
            local_path = Path(local_dir) / rel_path
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Download
            print(f"⬇️  Downloading s3://{bucket}/{key} → {local_path}")
            s3_client.download_file(bucket, key, str(local_path))

if __name__ == "__main__":
    config = Config()

    session = boto3.Session(profile_name=config.aws_profile) if config.aws_profile else boto3.Session()
    s3 = session.client("s3")

    bucket, prefix = parse_s3_uri(config.s3_uri)
    local_target = config.project_dir.parent / "Model"
    local_target.mkdir(exist_ok=True)

    download_s3_folder(s3, bucket, prefix, local_target)

    print(f"\nModel folder downloaded successfully to: {local_target.resolve()}")