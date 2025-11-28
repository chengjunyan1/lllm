"""
Minimal walkthrough that provisions a JupyterSandbox session, writes a notebook,
and highlights where artifacts are stored.

Usage:

    python examples/jupyter_sandbox_smoke.py --session demo

The script avoids executing cells so it runs safely in environments without a
local kernel. Inspect the printed directory afterwards to open the notebook
manually if desired.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lllm.sandbox.jupyter import JupyterSandbox


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JupyterSandbox smoke test")
    parser.add_argument(
        "--session",
        default="sandbox_demo",
        help="Name for the sandbox session/notebook.",
    )
    parser.add_argument(
        "--sandbox-dir",
        default=".lllm_sandbox",
        help="Folder where sandbox sessions should be created.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Path treated as the project root for proxy imports.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose sandbox logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    sandbox_root = Path(args.sandbox_dir).resolve()
    sandbox_root.mkdir(parents=True, exist_ok=True)

    config = {
        "name": "smoke",
        "project_root": project_root.as_posix(),
        "activate_proxies": [],
    }

    sandbox = JupyterSandbox(config, path=sandbox_root.as_posix(), verbose=args.verbose)
    session = sandbox.new_session(name=args.session)

    session.append_markdown_cell("# LLLM Sandbox Demo\nThis notebook was generated automatically.")
    session.append_code_cell("print('Hello from JupyterSandbox!')")
    session.shutdown_kernel()

    print(f"Notebook created at: {session.notebook_file}")
    print("Open it with Jupyter or VS Code to inspect the cells.")


if __name__ == "__main__":
    main()
