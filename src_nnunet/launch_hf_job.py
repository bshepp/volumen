# Pipeline: nnU-Net (src_nnunet/) — See PIPELINES.md
"""
Launch nnU-Net training on Hugging Face Jobs.

Uses direct HTTP requests to the HF API (avoids huggingface_hub SSL issues).

Usage:
    python -m src_nnunet.launch_hf_job --kaggle-token KGAT_xxx
    python -m src_nnunet.launch_hf_job --kaggle-username USER --kaggle-key KEY
"""

import argparse
import base64
import json
import os
import sys
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


TRAIN_SCRIPT = os.path.join(os.path.dirname(__file__), "train_hf.py")
DOCKER_IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
FLAVOR = "a10g-small"
TIMEOUT = "12h"
HF_API = "https://huggingface.co/api"
MODEL_REPO = "bshepp/vesuvius-nnunet-model"


def get_hf_token():
    for path in [
        os.path.expanduser("~/.cache/huggingface/token"),
        os.path.expanduser("~/.huggingface/token"),
    ]:
        if os.path.isfile(path):
            with open(path) as f:
                return f.read().strip()

    from huggingface_hub import HfFolder
    token = HfFolder.get_token()
    if token:
        return token

    raise RuntimeError("No HF token found. Run: huggingface-cli login")


def hf_session():
    """Create a requests session with automatic retries for transient SSL errors."""
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=1.0, status_forcelist=[502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


def hf_headers(token):
    return {"Authorization": f"Bearer {token}"}


def hf_request(method, url, token, max_attempts=5, **kwargs):
    """Make an HF API request with retry on SSL errors."""
    kwargs.setdefault("timeout", 30)
    headers = kwargs.pop("headers", {})
    headers["Authorization"] = f"Bearer {token}"

    for attempt in range(max_attempts):
        try:
            s = hf_session()
            r = getattr(s, method)(url, headers=headers, **kwargs)
            return r
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
            if attempt < max_attempts - 1:
                wait = 2 ** attempt
                print(f"  SSL error (attempt {attempt + 1}/{max_attempts}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def whoami(token):
    r = hf_request("get", f"{HF_API}/whoami-v2", token)
    r.raise_for_status()
    return r.json()


def create_repo(token, repo_id, repo_type="model"):
    namespace, name = repo_id.split("/", 1)
    r = hf_request(
        "post", f"{HF_API}/repos/create", token,
        json={"type": repo_type, "name": name, "organization": namespace, "private": False},
    )
    if r.status_code == 409:
        print(f"Repo {repo_id} already exists")
    elif r.status_code in (200, 201):
        print(f"Created repo: https://huggingface.co/{repo_id}")
    else:
        print(f"Repo creation: {r.status_code} — {r.text[:200]}")


def parse_timeout(timeout_str):
    """Convert timeout string like '12h' to seconds."""
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if isinstance(timeout_str, str) and timeout_str[-1] in units:
        return int(float(timeout_str[:-1]) * units[timeout_str[-1]])
    return int(timeout_str)


def run_job(token, namespace, image, command, flavor, timeout, secrets, env):
    payload = {
        "dockerImage": image,
        "command": command,
        "arguments": [],
        "environment": env or {},
        "flavor": flavor,
    }
    if secrets:
        payload["secrets"] = secrets
    if timeout:
        payload["timeoutSeconds"] = parse_timeout(timeout)

    r = hf_request(
        "post", f"{HF_API}/jobs/{namespace}", token,
        json=payload, timeout=60,
    )
    if not r.ok:
        print(f"Job launch failed: {r.status_code} — {r.text[:500]}")
    r.raise_for_status()
    return r.json()


def main():
    parser = argparse.ArgumentParser(description="Launch nnU-Net training on HF Jobs")
    auth_group = parser.add_mutually_exclusive_group(required=True)
    auth_group.add_argument("--kaggle-token", help="Kaggle API token (KGAT_... format)")
    auth_group.add_argument("--kaggle-username", help="Kaggle username (use with --kaggle-key)")
    parser.add_argument("--kaggle-key", help="Kaggle API key (use with --kaggle-username)")
    parser.add_argument("--flavor", default=FLAVOR,
                        help=f"GPU flavor (default: {FLAVOR})")
    parser.add_argument("--timeout", default=TIMEOUT,
                        help=f"Job timeout (default: {TIMEOUT})")
    parser.add_argument("--repo-id", default=MODEL_REPO,
                        help=f"HF model repo for saving model (default: {MODEL_REPO})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the job config without launching")
    args = parser.parse_args()

    token = get_hf_token()
    user_info = whoami(token)
    username = user_info["name"]
    print(f"Authenticated as: {username}")

    # Create model repo for saving results
    create_repo(token, args.repo_id)

    # Embed train_hf.py directly in the job command via base64
    with open(TRAIN_SCRIPT, "r") as f:
        script_content = f.read()
    script_b64 = base64.b64encode(script_content.encode()).decode()
    print(f"Training script: {len(script_content)} bytes → {len(script_b64)} base64 chars")

    # The job decodes the script, installs deps, and runs it
    bash_cmd = (
        f"echo '{script_b64}' | base64 -d > /tmp/train_hf.py && "
        "pip install -q nnunetv2 kaggle huggingface-hub tifffile imagecodecs && "
        "python -u /tmp/train_hf.py"
    )

    secrets = {"HF_TOKEN": token}
    if args.kaggle_token:
        secrets["KAGGLE_API_TOKEN"] = args.kaggle_token
    else:
        if not args.kaggle_key:
            parser.error("--kaggle-key is required when using --kaggle-username")
        secrets["KAGGLE_USERNAME"] = args.kaggle_username
        secrets["KAGGLE_KEY"] = args.kaggle_key

    env = {
        "HF_REPO_ID": args.repo_id,
        "HF_USERNAME": username,
    }

    if args.dry_run:
        print(f"\n--- DRY RUN ---")
        print(f"Image:   {DOCKER_IMAGE}")
        print(f"Flavor:  {args.flavor}")
        print(f"Timeout: {args.timeout}")
        print(f"Secrets: {', '.join(secrets.keys())}")
        print(f"Env:     {env}")
        print(f"\nEstimated cost: ~$8-10 for A10G, ~$4-5 for T4")
        print("\nRe-run without --dry-run to launch.")
        return

    print(f"\nLaunching job on {args.flavor} (timeout: {args.timeout})...")
    print(f"Estimated cost: ~$8-10 for 200 epochs on A10G\n")

    result = run_job(
        token=token,
        namespace=username,
        image=DOCKER_IMAGE,
        command=["bash", "-c", bash_cmd],
        flavor=args.flavor,
        timeout=args.timeout,
        secrets=secrets,
        env=env,
    )

    print("Job launched!")
    job_id = result.get("id", result.get("jobId", result.get("job_id", "unknown")))
    print(f"  Job ID: {job_id}")
    print(f"  Response: {json.dumps(result, indent=2)}")

    print(f"\nMonitor at: https://huggingface.co/settings/jobs")
    print(f"\nOr check logs:")
    print(f"  python -m src_nnunet.monitor_hf_job {job_id} --logs")


if __name__ == "__main__":
    main()
