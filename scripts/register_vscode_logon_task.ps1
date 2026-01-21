param(
    [string]$TaskName = "NullMove-Open-VSCode",
    [switch]$Remove
)

$ErrorActionPreference = 'Stop'

$Root = "D:\NullMove"
$Opener = Join-Path $Root "scripts\open_vscode.ps1"
if (-not (Test-Path $Opener)) {
    throw "Missing: $Opener"
}

if ($Remove) {
    schtasks /Delete /TN $TaskName /F | Out-Null
    Write-Host "Removed scheduled task: $TaskName"
    exit 0
}

$tr = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Opener`""

# Run at logon (more reliable than Startup folder when Startup apps are restricted).
schtasks /Create /F /TN $TaskName /SC ONLOGON /RL LIMITED /TR $tr | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to create scheduled task: $TaskName" -ForegroundColor Red
    Write-Host "Command: $tr"
    Write-Host "If you see 'ERROR: Access is denied.', run PowerShell as Administrator and re-run this script." -ForegroundColor Yellow
    exit $LASTEXITCODE
}

Write-Host "Created scheduled task: $TaskName"
Write-Host "Command: $tr"
