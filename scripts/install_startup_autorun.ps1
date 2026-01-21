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
# This opens VS Code on the workspace; the task in .vscode/tasks.json auto-starts training in the VS Code terminal.
$opener = Join-Path $Root "scripts\open_vscode.ps1"
if (-not (Test-Path $opener)) {
    throw "Missing: $opener"
}

$cmd = @(
"@echo off",
"powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$opener`""
) -join "`r`n"

Set-Content -Path $cmdPath -Value $cmd -Encoding ASCII
Write-Host "Installed startup autorun: $cmdPath"
Write-Host "It will open VS Code after you log in."
Write-Host "Training will start in the VS Code terminal via the auto-run task."
