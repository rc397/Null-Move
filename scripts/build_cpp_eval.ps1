param(
  [string]$OutDir = "D:/NullMove/bin",
  [string]$Src = "D:/NullMove/cpp/nullmove_eval.cpp"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

$cl = Get-Command cl.exe -ErrorAction SilentlyContinue
if (-not $cl) {
  Write-Host "cl.exe not found. Install 'Visual Studio Build Tools' (C++ workload) and re-open a Developer PowerShell." -ForegroundColor Yellow
  Write-Host "Source is at: $Src"
  exit 1
}

Push-Location $OutDir
try {
  & cl.exe /O2 /LD /EHsc $Src /Fe:nullmove_eval.dll
  Write-Host "Built $OutDir/nullmove_eval.dll" -ForegroundColor Green
} finally {
  Pop-Location
}
