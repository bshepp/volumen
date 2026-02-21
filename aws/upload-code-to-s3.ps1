# Upload code to S3 so the next EC2 run (or manual sync on instance) gets it.
# Run from repo root:  .\aws\upload-code-to-s3.ps1
# Requires: AWS CLI configured, bucket vesuvius-challenge-training-290318

$ErrorActionPreference = "Stop"
$Bucket = "vesuvius-challenge-training-290318"
$CodePrefix = "code"

$RepoRoot = Split-Path $PSScriptRoot -Parent
if (-not (Test-Path (Join-Path $RepoRoot "src"))) {
    Write-Error "Run from repo root or ensure src/ exists. Repo root used: $RepoRoot"
}

Push-Location $RepoRoot

Write-Host "Uploading code to s3://$Bucket/$CodePrefix/ ..."
# Sync src/ (Pipeline V1); exclude bytecode
aws s3 sync src/ "s3://$Bucket/$CodePrefix/src/" --delete --exclude "__pycache__/*" --exclude "*.pyc"
# Sync src_v2/ (Pipeline V2 - for post-run start)
aws s3 sync src_v2/ "s3://$Bucket/$CodePrefix/src_v2/" --delete --exclude "__pycache__/*" --exclude "*.pyc"
# requirements.txt (includes scikit-image for V2)
aws s3 cp requirements.txt "s3://$Bucket/$CodePrefix/requirements.txt"
# aws/*.sh so instance can run post-run script after V1 completes
aws s3 sync aws/ "s3://$Bucket/$CodePrefix/aws/" --exclude "*.json" --exclude "*.ps1" --exclude "*.md"

Write-Host "Done. New runs will pull this code via user-data."
Write-Host "After V1 completes, run: .\aws\post-run-download-and-start-v2.ps1"
Pop-Location
