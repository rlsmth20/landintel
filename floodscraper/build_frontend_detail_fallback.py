from __future__ import annotations

import argparse
import runpy
import sys


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=True, description="Run the frontend detail-fallback builder through a generic entrypoint.")
    parser.add_argument("--state-code", default="ms")
    return parser.parse_known_args()


def main() -> None:
    args, passthrough = parse_args()
    state_code = str(args.state_code).strip().lower()
    if state_code != "ms":
        raise NotImplementedError(f"Frontend fallback generation is only implemented for state_code={state_code!r} at this stage.")
    sys.argv = ["build_frontend_detail_fallback_ms.py", *passthrough]
    runpy.run_module("build_frontend_detail_fallback_ms", run_name="__main__")


if __name__ == "__main__":
    main()
