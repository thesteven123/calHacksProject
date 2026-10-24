"""Verify HF_TOKEN and TINKER_API_KEY with read-only calls. Never prints secret values or account names.

    .venv-tinker/bin/python scripts/check_credentials.py
"""

from __future__ import annotations

import os
import sys

from dfd.env import load_dotenv

GATED = (
    ("model", "facebook/dinov3-vitb16-pretrain-lvd1689m"),
    ("dataset", "ControlNet/AV-Deepfake1M"),
)


def check_hf() -> bool:
    if not os.environ.get("HF_TOKEN"):
        print("HF_TOKEN: missing")
        return False
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    try:
        api.whoami()
    except Exception as error:  # invalid/expired token
        print(f"HF_TOKEN: rejected ({type(error).__name__})")
        return False
    print("HF_TOKEN: valid (authenticated)")
    for kind, repo in GATED:
        try:
            api.auth_check(repo, repo_type=kind)
            print(f"  gated {kind} {repo}: access granted")
        except Exception as error:
            print(f"  gated {kind} {repo}: no access yet ({type(error).__name__}) - request/await approval")
    return True


def check_tinker() -> bool:
    if not os.environ.get("TINKER_API_KEY"):
        print("TINKER_API_KEY: missing")
        return False
    try:
        import tinker
    except ImportError:
        print("TINKER_API_KEY: present, but `tinker` is not installed in this interpreter (use .venv-tinker)")
        return False
    try:
        caps = tinker.ServiceClient(api_key=os.environ["TINKER_API_KEY"]).get_server_capabilities()
    except Exception as error:
        print(f"TINKER_API_KEY: rejected or unreachable ({type(error).__name__})")
        return False
    names = [getattr(m, "model_name", str(m)) for m in caps.supported_models]
    vl = [n for n in names if "VL" in n or "Qwen3.5" in n or "Qwen3.6" in n]
    print(f"TINKER_API_KEY: valid ({len(names)} models available, {len(vl)} vision-capable candidates)")
    return True


if __name__ == "__main__":
    loaded = load_dotenv()
    if loaded:
        print(f".env loaded: {', '.join(sorted(loaded))}")
    ok = [check_hf(), check_tinker()]
    sys.exit(0 if all(ok) else 1)
