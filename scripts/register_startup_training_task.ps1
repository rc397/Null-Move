param(
    [string]$TaskName = "NullMove-Training",
    [switch]$Remove,
    [switch]$OpenVSCode,
    [switch]$Highest
)

$ErrorActionPreference = 'Stop'

$Root = "D:\NullMove"
$Runner = Join-Path $Root "scripts\start_training.ps1"

if ($Remove) {
    schtasks /Delete /TN $TaskName /F | Out-Null
    Write-Host "Removed scheduled task: $TaskName"
    exit 0
}

if (-not (Test-Path $Runner)) {
    throw "Runner script not found: $Runner"
}

# Build the task command.
# Important: use -ExecutionPolicy Bypass so it runs unattended.
$openArg = ""
if ($OpenVSCode) { $openArg = " -OpenVSCode" }

$tr = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Runner`"$openArg -NoTail"

# Create/replace a task that runs at logon for the current user.
# (Runs when you boot + log in. Startup-without-login usually requires admin/service.)

$rl = "LIMITED"
if ($Highest) { $rl = "HIGHEST" }

try {
    schtasks /Create /F /TN $TaskName /SC ONLOGON /RL $rl /TR $tr | Out-Null
    Write-Host "Created scheduled task: $TaskName"
    Write-Host "It will run on logon. You can view it in Task Scheduler." 
    Write-Host "Command: $tr"
} catch {
    Write-Host "Failed to create Scheduled Task (policy/permissions)."
    Write-Host "Fallback (no admin needed):"
    Write-Host "  powershell -ExecutionPolicy Bypass -File D:/NullMove/scripts/install_startup_autorun.ps1"
    throw
}
