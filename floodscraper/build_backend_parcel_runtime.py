from __future__ import annotations

import argparse
import runpy
import sys


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=True, description="Run the state runtime builder through a generic entrypoint.")
    parser.add_argument("--state-code", default="ms")
    return parser.parse_known_args()


def main() -> None:
    args, passthrough = parse_args()
    state_code = str(args.state_code).strip().lower()
    module_name = {
        "ar": "build_backend_parcel_runtime_ar",
        "ms": "build_backend_parcel_runtime_ms",
    }.get(state_code, "build_backend_parcel_runtime_arcgis")
    sys.argv = [f"{module_name}.py", *passthrough]
    if module_name == "build_backend_parcel_runtime_arcgis":
        sys.argv.extend(["--state-code", state_code])
    runpy.run_module(module_name, run_name="__main__")


if __name__ == "__main__":
    main()
