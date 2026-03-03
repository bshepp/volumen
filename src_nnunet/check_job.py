# Quick script to check HF Job logs
import requests, os, json, sys

for p in [os.path.expanduser("~/.cache/huggingface/token"),
          os.path.expanduser("~/.huggingface/token")]:
    if os.path.isfile(p):
        with open(p) as f:
            token = f.read().strip()
        break
else:
    from huggingface_hub import HfFolder
    token = HfFolder.get_token()

JOB_ID = "69a6403edfb316ac3f7c2117"
headers = {"Authorization": f"Bearer {token}"}

# Status
r = requests.get(f"https://huggingface.co/api/jobs/bshepp/{JOB_ID}", headers=headers, timeout=30)
info = r.json()
print(f"Status: {info['status']['stage']}")
msg = info["status"].get("message")
if msg:
    print(f"Message: {msg}")
print()

# Logs (streaming)
r = requests.get(
    f"https://huggingface.co/api/jobs/bshepp/{JOB_ID}/logs",
    headers=headers, timeout=15, stream=True,
)
lines = []
try:
    for line in r.iter_lines(decode_unicode=True):
        if line and line.startswith("data:"):
            payload = line[5:].strip()
            if payload:
                data = json.loads(payload)
                lines.append(data.get("data", ""))
                if len(lines) > 1000:
                    break
except Exception:
    pass

n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
for line in lines[-n:]:
    print(line)
print(f"\n--- {len(lines)} total log lines ---")
