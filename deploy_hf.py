#!/usr/bin/env python3
"""Deploy this project to a Hugging Face Space."""

import os
import subprocess
import sys

from huggingface_hub import HfApi


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def main() -> None:
    token = require_env("HF_TOKEN")
    space_name = os.getenv("HF_SPACE_NAME", "who-spoke-when")

    api = HfApi(token=token)

    username = os.getenv("HF_USERNAME")
    if not username:
        whoami = api.whoami(token=token)
        username = whoami["name"]

    space_id = f"{username}/{space_name}"

    try:
        api.create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="docker",
            private=False,
            token=token,
            exist_ok=True,
        )
        print(f"Space ready: {space_id}")
    except Exception as exc:
        raise SystemExit(f"Failed to create or fetch space '{space_id}': {exc}") from exc

    remote_url = f"https://{username}:{token}@huggingface.co/spaces/{space_id}"
    subprocess.run(["git", "remote", "remove", "huggingface"], check=False, capture_output=True)
    subprocess.run(["git", "remote", "add", "huggingface", remote_url], check=True)

    push_cmd = ["git", "push", "huggingface", "main"]
    if os.getenv("HF_FORCE_PUSH", "false").lower() in {"1", "true", "yes"}:
        push_cmd.append("--force")

    subprocess.run(push_cmd, check=True)
    print(f"Pushed to https://huggingface.co/spaces/{space_id}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
