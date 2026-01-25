param(
    [string]$TaskPrefix = "NullMove",

    # Daily window
    [string]$StartTime = "07:00",
    [string]$StopTime = "21:00",

    # If true, the start task attempts to wake the PC (requires Windows wake timers enabled)
    [switch]$WakeToRun,

    # If true, open VS Code at start (training then starts via VS Code folder-open task)
    [switch]$OpenVSCode,

    # Remove tasks instead of creating
    [switch]$Remove
)

$ErrorActionPreference = 'Stop'

$Root = "D:\NullMove"
$StartTask = "$TaskPrefix-Train-Start"
$StopTask = "$TaskPrefix-Train-Stop"

$startCmd = if ($OpenVSCode) {
    # Single canonical entrypoint; optionally opens VS Code too.
    "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Root\scripts\start_training.ps1`" -OpenVSCode -NoTail"
} else {
    # Starts training directly (more reliable than VS Code integration)
    "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Root\scripts\start_training.ps1`" -NoTail"
}

$stopCmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Root\scripts\stop_training.ps1`""

if ($Remove) {
    schtasks /Delete /TN $StartTask /F | Out-Null
    schtasks /Delete /TN $StopTask /F | Out-Null
    Write-Host "Removed tasks: $StartTask and $StopTask"
    exit 0
}

# Create start/stop tasks.
# NOTE: Some systems restrict schtasks creation via policy; if you see Access Denied, you'll need to create them manually in Task Scheduler.

schtasks /Create /F /TN $StartTask /SC DAILY /ST $StartTime /TR $startCmd | Out-Null
schtasks /Create /F /TN $StopTask /SC DAILY /ST $StopTime /TR $stopCmd | Out-Null

Write-Host "Created tasks:"
Write-Host "- $StartTask at $StartTime"
Write-Host "- $StopTask at $StopTime"

if ($WakeToRun) {
    Write-Host "Wake-to-run must be enabled in Task Scheduler UI (schtasks.exe cannot reliably set it without XML)."
    Write-Host "Open Task Scheduler -> $StartTask -> Properties -> Conditions -> check 'Wake the computer to run this task'."
}

Write-Host "Tip: If you want the PC to 'boot itself' at 7am from OFF, set a BIOS/UEFI RTC wake timer."
Write-Host "If you only need wake-from-sleep, enable wake timers in Windows Power Options."
