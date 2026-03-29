from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from state_artifacts import load_state_artifacts
from state_registry import ROOT


WRANGLER_R2_MAX_UPLOAD_BYTES = 300 * 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a state parcel PMTiles archive to Cloudflare R2.")
    parser.add_argument("--state-code", required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--bucket", default=None, help="Cloudflare R2 bucket name. Falls back to CLOUDFLARE_R2_BUCKET.")
    parser.add_argument("--public-base-url", default=None, help="Optional public base URL used to print the final asset URL.")
    parser.add_argument(
        "--transport",
        choices=("auto", "wrangler", "boto3"),
        default="auto",
        help="Upload transport. 'auto' prefers boto3 when an R2 endpoint is available and falls back to Wrangler.",
    )
    parser.add_argument(
        "--endpoint-url",
        default=None,
        help="S3-compatible Cloudflare R2 endpoint URL. Falls back to CLOUDFLARE_R2_ENDPOINT_URL or AWS_ENDPOINT_URL.",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="S3 region used for boto3 uploads. Falls back to CLOUDFLARE_R2_REGION or 'auto'.",
    )
    parser.add_argument("--wrangler-bin", default="wrangler", help="Wrangler executable to call when --execute is supplied.")
    parser.add_argument("--execute", action="store_true", help="Run the upload command instead of printing it.")
    return parser.parse_args()


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_path_for_state(state_code: str, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path
    artifacts = load_state_artifacts(state_code)
    legacy_runtime_dir = artifacts.runtime_summary_path.parent
    return legacy_runtime_dir / "parcel_pmtiles_publish_manifest.json"


def build_wrangler_command(
    manifest: dict[str, Any],
    *,
    bucket: str,
    wrangler_bin: str,
) -> list[str]:
    return [
        wrangler_bin,
        "r2",
        "object",
        "put",
        f"{bucket}/{manifest['cloudflare_object_key']}",
        "--remote",
        "--file",
        str(Path(manifest["artifact_path"])),
        "--content-type",
        str(manifest["content_type"]),
        "--cache-control",
        str(manifest["cache_control"]),
    ]


def _quote_command(command: list[str]) -> str:
    return " ".join(shlex.quote(segment) for segment in command)


def _resolve_wrangler_bin(wrangler_bin: str) -> str:
    resolved = shutil.which(wrangler_bin)
    if resolved:
        return resolved
    if sys.platform.startswith("win") and "." not in Path(wrangler_bin).name:
        for candidate in ("wrangler.cmd", "wrangler.exe", "wrangler.ps1"):
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
    return wrangler_bin


def resolve_endpoint_url(explicit_endpoint_url: str | None) -> str | None:
    return explicit_endpoint_url or os.environ.get("CLOUDFLARE_R2_ENDPOINT_URL") or os.environ.get("AWS_ENDPOINT_URL")


def resolve_region(explicit_region: str | None) -> str:
    return explicit_region or os.environ.get("CLOUDFLARE_R2_REGION") or "auto"


def select_transport(
    manifest: dict[str, Any],
    *,
    requested_transport: str,
    endpoint_url: str | None,
) -> str:
    if requested_transport != "auto":
        return requested_transport
    if endpoint_url:
        return "boto3"
    artifact_path = Path(manifest["artifact_path"])
    artifact_size = artifact_path.stat().st_size if artifact_path.exists() else 0
    if artifact_size > WRANGLER_R2_MAX_UPLOAD_BYTES:
        return "boto3"
    return "wrangler"


def _execute_boto3_upload(
    manifest: dict[str, Any],
    *,
    bucket: str,
    endpoint_url: str,
    region: str,
) -> None:
    try:
        import boto3
        from boto3.s3.transfer import TransferConfig
        from botocore.config import Config
    except ImportError as exc:  # pragma: no cover - exercised via integration use
        raise SystemExit(
            "boto3 is required for --transport boto3. Install it with 'python -m pip install boto3'."
        ) from exc

    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=region,
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 5, "mode": "standard"},
        ),
    )
    transfer_config = TransferConfig(
        max_concurrency=1,
        multipart_threshold=8 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        use_threads=False,
    )
    client.upload_file(
        str(Path(manifest["artifact_path"])),
        bucket,
        str(manifest["cloudflare_object_key"]),
        ExtraArgs={
            "ContentType": str(manifest["content_type"]),
            "CacheControl": str(manifest["cache_control"]),
        },
        Config=transfer_config,
    )


def main() -> None:
    args = parse_args()
    manifest_path = _manifest_path_for_state(args.state_code, args.manifest)
    manifest = _load_manifest(manifest_path)
    bucket = args.bucket or __import__("os").environ.get("CLOUDFLARE_R2_BUCKET")
    if not bucket:
        raise SystemExit("Cloudflare R2 bucket not provided. Use --bucket or set CLOUDFLARE_R2_BUCKET.")

    endpoint_url = resolve_endpoint_url(args.endpoint_url)
    region = resolve_region(args.region)
    transport = select_transport(
        manifest,
        requested_transport=args.transport,
        endpoint_url=endpoint_url,
    )
    if transport == "boto3" and not endpoint_url:
        raise SystemExit(
            "Cloudflare R2 endpoint URL not provided for boto3 upload. Use --endpoint-url or set CLOUDFLARE_R2_ENDPOINT_URL."
        )

    wrangler_bin = _resolve_wrangler_bin(args.wrangler_bin)
    command = None
    command_shell = None
    if transport == "wrangler":
        command = build_wrangler_command(manifest, bucket=bucket, wrangler_bin=wrangler_bin)
        command_shell = _quote_command(command)
    configured_public_url = manifest.get("public_url")
    public_url = configured_public_url
    if args.public_base_url:
        public_url = args.public_base_url.rstrip("/") + "/" + str(manifest["cloudflare_object_key"]).lstrip("/")

    output = {
        "state_code": args.state_code,
        "manifest_path": str(manifest_path),
        "bucket": bucket,
        "transport": transport,
        "endpoint_url": endpoint_url,
        "region": region,
        "command": command,
        "command_shell": command_shell,
        "configured_public_url": configured_public_url,
        "public_url": public_url,
        "executed": bool(args.execute),
    }

    if args.execute:
        if transport == "wrangler":
            assert command is not None
            subprocess.run(command, cwd=ROOT, check=True)
        else:
            assert endpoint_url is not None
            _execute_boto3_upload(
                manifest,
                bucket=bucket,
                endpoint_url=endpoint_url,
                region=region,
            )

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
