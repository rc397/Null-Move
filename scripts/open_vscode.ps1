param(
    [string]$Workspace = "D:\\NullMove"
)

$ErrorActionPreference = 'Continue'

$LogDir = Join-Path $Workspace "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogPath = Join-Path $LogDir "open_vscode.log"

"[$(Get-Date -Format o)] open_vscode.ps1 starting" | Add-Content -Path $LogPath -Encoding UTF8
"[$(Get-Date -Format o)] Workspace=$Workspace" | Add-Content -Path $LogPath -Encoding UTF8

try {
    $code = Get-Command code -ErrorAction SilentlyContinue
    if ($null -ne $code) {
        "[$(Get-Date -Format o)] Using PATH code: $($code.Source)" | Add-Content -Path $LogPath -Encoding UTF8
        Start-Process -FilePath $code.Source -ArgumentList @('--reuse-window', $Workspace) | Out-Null
        "[$(Get-Date -Format o)] VS Code started via PATH" | Add-Content -Path $LogPath -Encoding UTF8
        exit 0
    }

    $candidates = @(
        "$env:LOCALAPPDATA\\Programs\\Microsoft VS Code\\Code.exe",
        "$env:ProgramFiles\\Microsoft VS Code\\Code.exe",
        "${env:ProgramFiles(x86)}\\Microsoft VS Code\\Code.exe"
    )

    foreach ($p in $candidates) {
        if (Test-Path $p) {
            "[$(Get-Date -Format o)] Using Code.exe: $p" | Add-Content -Path $LogPath -Encoding UTF8
            Start-Process -FilePath $p -ArgumentList @('--reuse-window', $Workspace) | Out-Null
            "[$(Get-Date -Format o)] VS Code started via direct path" | Add-Content -Path $LogPath -Encoding UTF8
            exit 0
        }
    }

    "[$(Get-Date -Format o)] ERROR: VS Code not found" | Add-Content -Path $LogPath -Encoding UTF8
    exit 2
} catch {
    "[$(Get-Date -Format o)] ERROR: $($_.Exception.Message)" | Add-Content -Path $LogPath -Encoding UTF8
    exit 1
}
