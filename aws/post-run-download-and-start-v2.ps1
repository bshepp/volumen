# After V1 run completes (epoch 200): verify, run post-run script on instance, download best model.
# Usage: .\aws\post-run-download-and-start-v2.ps1 [-InstanceId i-xxx] [-SkipDownload]
# Requires: AWS CLI, instance ID (default from our training instance).

param(
    [string]$InstanceId = "i-00035acc764c2bd27",
    [switch]$SkipDownload
)

$Bucket = "vesuvius-challenge-training-290318"
$RepoRoot = Split-Path $PSScriptRoot -Parent
$OutputsAws = Join-Path $RepoRoot "outputs_aws"

Write-Host "Instance: $InstanceId"
Write-Host ""

# 1) Check if V1 training has completed
Write-Host "Checking if V1 training has completed..."
Push-Location $RepoRoot
$cid = aws ssm send-command --instance-ids $InstanceId --document-name "AWS-RunShellScript" --parameters "file://aws/ssm-check-complete.json" --query "Command.CommandId" --output text 2>$null
if (-not $cid) {
    Pop-Location
    Write-Host "Could not send SSM command. Is the instance running and has SSM agent?"
    exit 1
}
Start-Sleep -Seconds 6
$result = aws ssm get-command-invocation --command-id $cid --instance-id $InstanceId --query "StandardOutputContent" --output text 2>$null
Pop-Location
$lastLines = $result -split "`n"
$hasComplete = ($result -match "1" -or $result -match "Training complete")
if (-not $hasComplete) {
    Write-Host "Last lines of training.log:"
    Write-Host $result
    Write-Host ""
    $r = Read-Host "Training may not be complete. Run post-run script anyway? (y/n)"
    if ($r -ne "y") { exit 0 }
}

# 2) Run post-run script on instance (code/aws/ must be on S3 - run upload-code-to-s3.ps1 first)
Write-Host "Running post-run script on instance (upload best to S3, clear, start V2)..."
Push-Location $RepoRoot
$cid2 = aws ssm send-command --instance-ids $InstanceId --document-name "AWS-RunShellScript" --parameters "file://aws/ssm-postrun.json" --timeout-seconds 300 --query "Command.CommandId" --output text
Pop-Location
Start-Sleep -Seconds 15
$inv = aws ssm get-command-invocation --command-id $cid2 --instance-id $InstanceId --output json 2>$null | ConvertFrom-Json
Write-Host "Status: $($inv.Status)"
Write-Host $inv.StandardOutputContent
if ($inv.StandardErrorContent) { Write-Host "Stderr: $($inv.StandardErrorContent)" }
if ($inv.Status -eq "Failed" -or $inv.Status -eq "Cancelled") {
    Write-Host "Post-run script failed. Fix and re-run manually if needed."
    exit 1
}

# 3) Download best model from S3 to local
if (-not $SkipDownload) {
    New-Item -ItemType Directory -Force -Path $OutputsAws | Out-Null
    Write-Host ""
    Write-Host "Downloading run1_best_model.pth from S3 to $OutputsAws ..."
    aws s3 cp "s3://$Bucket/outputs/run1_best_model.pth" (Join-Path $OutputsAws "run1_best_model.pth") --region us-east-1
    aws s3 cp "s3://$Bucket/outputs/run1_history.json" (Join-Path $OutputsAws "run1_history.json") --region us-east-1 2>$null
    aws s3 cp "s3://$Bucket/outputs/run1_training.log" (Join-Path $OutputsAws "run1_training.log") --region us-east-1 2>$null
    Write-Host "Done. Best V1 model: $OutputsAws\run1_best_model.pth"
}

Write-Host ""
Write-Host "V2 training should now be running on the instance. Check log: outputs_v2/training.log (via SSM or SSH)."
