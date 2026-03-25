from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

from state_artifacts import load_state_artifacts
from state_registry import ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a state parcel PMTiles archive to Cloudflare R2 via Wrangler.")
    parser.add_argument("--state-code", required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--bucket", default=None, help="Cloudflare R2 bucket name. Falls back to CLOUDFLARE_R2_BUCKET.")
    parser.add_argument("--public-base-url", default=None, help="Optional public base URL used to print the final asset URL.")
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
        "--file",
        str(Path(manifest["artifact_path"])),
        "--content-type",
        str(manifest["content_type"]),
        "--cache-control",
        str(manifest["cache_control"]),
    ]


def _quote_command(command: list[str]) -> str:
    return " ".join(shlex.quote(segment) for segment in command)


def main() -> None:
    args = parse_args()
    manifest_path = _manifest_path_for_state(args.state_code, args.manifest)
    manifest = _load_manifest(manifest_path)
    bucket = args.bucket or __import__("os").environ.get("CLOUDFLARE_R2_BUCKET")
    if not bucket:
        raise SystemExit("Cloudflare R2 bucket not provided. Use --bucket or set CLOUDFLARE_R2_BUCKET.")

    command = build_wrangler_command(manifest, bucket=bucket, wrangler_bin=args.wrangler_bin)
    public_url = None
    if args.public_base_url:
        public_url = args.public_base_url.rstrip("/") + "/" + str(manifest["cloudflare_object_key"]).lstrip("/")

    output = {
        "state_code": args.state_code,
        "manifest_path": str(manifest_path),
        "bucket": bucket,
        "command": command,
        "command_shell": _quote_command(command),
        "public_url": public_url,
        "executed": bool(args.execute),
    }

    if args.execute:
        subprocess.run(command, cwd=ROOT, check=True)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
