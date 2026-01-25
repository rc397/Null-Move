param(
    [switch]$Remove
)

$ErrorActionPreference = 'Stop'

$Root = "D:\NullMove"
$CodeCmd = "code"

$startup = [Environment]::GetFolderPath('Startup')
$cmdPath = Join-Path $startup "NullMove-Training.cmd"

if ($Remove) {
    if (Test-Path $cmdPath) {
        Remove-Item -Force $cmdPath
        Write-Host "Removed: $cmdPath"
    } else {
        Write-Host "Not found: $cmdPath"
    }
    exit 0
}

# Use Hidden to avoid an annoying console window every login.
# Use the same launcher as manual runs, so startup and manual behavior match.
$runner = Join-Path $Root "scripts\start_training.ps1"
if (-not (Test-Path $runner)) {
    throw "Missing: $runner"
}

$cmd = @(
"@echo off",
"powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$runner`" -OpenVSCode -NoTail"
) -join "`r`n"

Set-Content -Path $cmdPath -Value $cmd -Encoding ASCII
Write-Host "Installed startup autorun: $cmdPath"
Write-Host "It will start training after you log in."
Write-Host "(VS Code will open too, because -OpenVSCode is enabled.)"
