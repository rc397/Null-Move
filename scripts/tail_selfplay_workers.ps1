param(
    [int]$Workers = 6,
    [string]$LogsDir = "D:\NullMove\logs",
    [string]$BaseName = "selfplay_live",
    [switch]$NoCombined
)

$ErrorActionPreference = 'Stop'

if ($Workers -lt 1) {
    throw "Workers must be >= 1"
}

$logsPath = [System.IO.Path]::GetFullPath($LogsDir)
if (-not (Test-Path $logsPath)) {
    New-Item -ItemType Directory -Force -Path $logsPath | Out-Null
}

$combined = Join-Path $logsPath ("{0}.log" -f $BaseName)
New-Item -ItemType File -Force -Path $combined | Out-Null

function Start-TailWindow([string]$title, [string]$path) {
    New-Item -ItemType File -Force -Path $path | Out-Null
    $escapedPath = $path.Replace("'", "''")
    $escapedTitle = $title.Replace("'", "''")

    $cmd = "`$host.UI.RawUI.WindowTitle = '{0}'; Get-Content -Path '{1}' -Wait" -f $escapedTitle, $escapedPath
    Start-Process -FilePath "powershell.exe" -ArgumentList @('-NoExit', '-NoProfile', '-Command', $cmd) | Out-Null
}

if (-not $NoCombined) {
    Start-TailWindow -title "selfplay (combined)" -path $combined
}

for ($i = 0; $i -lt $Workers; $i++) {
    $workerPath = Join-Path $logsPath ("{0}_w{1}.log" -f $BaseName, $i)
    Start-TailWindow -title ("selfplay w{0}" -f $i) -path $workerPath
}

Write-Host "Opened tails for $Workers worker logs in $logsPath"
if (-not $NoCombined) {
    Write-Host "Combined log: $combined"
}
