# Pipeline: nnU-Net (src_nnunet/) — See PIPELINES.md
"""
Monitor a running Hugging Face Job.

Usage:
    python -m src_nnunet.monitor_hf_job JOB_ID
    python -m src_nnunet.monitor_hf_job JOB_ID --logs
    python -m src_nnunet.monitor_hf_job --list
"""

import argparse
import json
import os
import sys

import requests


HF_API = "https://huggingface.co/api"


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


def hf_headers(token):
    return {"Authorization": f"Bearer {token}"}


def main():
    parser = argparse.ArgumentParser(description="Monitor HF Jobs")
    parser.add_argument("job_id", nargs="?", help="Job ID to inspect")
    parser.add_argument("--logs", action="store_true", help="Fetch job logs")
    parser.add_argument("--list", action="store_true", help="List recent jobs")
    args = parser.parse_args()

    token = get_hf_token()

    if args.list:
        r = requests.get(f"{HF_API}/jobs", headers=hf_headers(token), timeout=30)
        r.raise_for_status()
        jobs = r.json()
        if not jobs:
            print("No jobs found.")
            return
        for job in jobs:
            jid = job.get("jobId", job.get("id", "?"))
            status = job.get("status", "?")
            flavor = job.get("flavor", "?")
            print(f"  {jid}  {status:<15}  {flavor}")
        return

    if not args.job_id:
        parser.print_help()
        sys.exit(1)

    if args.logs:
        r = requests.get(
            f"{HF_API}/jobs/{args.job_id}/logs",
            headers=hf_headers(token),
            timeout=30,
        )
        r.raise_for_status()
        print(r.text)
    else:
        r = requests.get(
            f"{HF_API}/jobs/{args.job_id}",
            headers=hf_headers(token),
            timeout=30,
        )
        r.raise_for_status()
        info = r.json()
        print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
