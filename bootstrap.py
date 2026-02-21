#!/usr/bin/env python3
"""
bootstrap.py

Create a local virtualenv and install dependencies without requiring `uv`.

Usage:
  python bootstrap.py            # runtime deps only
  python bootstrap.py --dev      # include pytest/ruff
  python bootstrap.py --recreate # delete and recreate .venv
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
import venv


RUNTIME_DEPS = [
    "pydantic>=2.0",
    "PyYAML>=6.0",
    "networkx>=3.0",
    "typer>=0.9",
    "rich>=13.0",
]

DEV_DEPS = [
    "pytest>=8.0",
    "ruff>=0.5",
]


def run(cmd: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=str(cwd), check=check)


def venv_python(venv_dir: Path) -> Path:
    if platform.system().lower().startswith("win"):
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true", help="install dev tools (pytest/ruff)")
    ap.add_argument("--recreate", action="store_true", help="delete and recreate .venv")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent
    venv_dir = repo / ".venv"

    if args.recreate and venv_dir.exists():
        shutil.rmtree(venv_dir)

    if not venv_dir.exists():
        print(f"[bootstrap] Creating venv at: {venv_dir}")
        venv.EnvBuilder(with_pip=True, clear=False).create(venv_dir)

    py = venv_python(venv_dir)
    if not py.exists():
        print(f"[bootstrap] ERROR: venv python not found at {py}")
        return 1

    print("[bootstrap] Upgrading pip/setuptools/wheel...")
    run([str(py), "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"], cwd=repo)

    print("[bootstrap] Installing project (editable) if possible...")
    ok = True
    try:
        run([str(py), "-m", "pip", "install", "-e", "."], cwd=repo)
    except subprocess.CalledProcessError:
        ok = False

    if not ok:
        print("[bootstrap] Project packaging not available; falling back to minimal deps install.")
        run([str(py), "-m", "pip", "install", *RUNTIME_DEPS], cwd=repo)

    if args.dev:
        print("[bootstrap] Installing dev deps...")
        run([str(py), "-m", "pip", "install", *DEV_DEPS], cwd=repo)

    print("[bootstrap] Sanity-check import...")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    try:
        subprocess.run(
            [str(py), "-c", "import semantic_opinion_dynamics; print('import OK')"],
            cwd=str(repo),
            check=True,
            env=env,
        )
    except subprocess.CalledProcessError:
        print("[bootstrap] WARNING: import failed. If you run the engine, use PYTHONPATH=src.")
    print()

    if platform.system().lower().startswith("win"):
        act = r".\.venv\Scripts\activate"
    else:
        act = "source .venv/bin/activate"

    print("Next steps:")
    print(f"  1) Activate venv: {act}")
    print("  2) Run simulation (no uv required):")
    print("       PYTHONPATH=src python -m semantic_opinion_dynamics.cli run \\")
    print("         --config config.yaml --network networks/example_org.json --personas personas.yaml")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())