"""Minimal .env loader so scripts pick up TINKER_API_KEY / HF_TOKEN without manual `source`.

Never overrides non-empty variables already set in the environment, and never prints values.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: str | Path | None = None) -> list[str]:
    """Load KEY=VALUE lines from `.env` (project root by default). Returns the names that were set."""
    env_path = Path(path) if path else PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return []
    loaded = []
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.removeprefix("export ").split("=", 1)
        name, value = name.strip(), value.strip().strip("\"'")
        if name and value and not os.environ.get(name):  # empty exported values count as unset
            os.environ[name] = value
            loaded.append(name)
    return loaded
