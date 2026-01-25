param(
    [string]$Root = "D:\NullMove",

    # Where parallel self-play leaves its chunk files (jsonl/pgn) if interrupted before merge.
    [string]$TmpDir = "",

    # Outputs to merge into.
    [string]$Dataset = "",
    [string]$PgnOut = "",
    [string]$WinsPgnOut = "",

    [switch]$WhatIf
)

$ErrorActionPreference = 'Stop'

if ([string]::IsNullOrWhiteSpace($TmpDir)) {
    $TmpDir = Join-Path $Root "_selfplay_tmp"
}
if ([string]::IsNullOrWhiteSpace($Dataset)) {
    $Dataset = Join-Path $Root "az_dataset.jsonl"
}
if ([string]::IsNullOrWhiteSpace($PgnOut)) {
    $PgnOut = Join-Path (Join-Path $Root "Self_Play_Games") "az_progress.pgn"
}
if ([string]::IsNullOrWhiteSpace($WinsPgnOut)) {
    $WinsPgnOut = Join-Path (Join-Path $Root "Self_Play_Games") "az_progress_wins.pgn"
}

function Ensure-ParentDir([string]$Path) {
    $parent = Split-Path -Parent $Path
    if (-not [string]::IsNullOrWhiteSpace($parent)) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
}

function Append-File([string]$Source, [string]$Dest) {
    Ensure-ParentDir $Dest

    if ($WhatIf) {
        Write-Host "[WhatIf] Append $Source -> $Dest"
        return
    }

    $inStream = [System.IO.File]::Open($Source, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read)
    try {
        $outStream = [System.IO.File]::Open($Dest, [System.IO.FileMode]::Append, [System.IO.FileAccess]::Write, [System.IO.FileShare]::Read)
        try {
            $inStream.CopyTo($outStream)
        } finally {
            $outStream.Dispose()
        }
    } finally {
        $inStream.Dispose()
    }
}

function Remove-FileSafe([string]$Path) {
    if ($WhatIf) {
        Write-Host "[WhatIf] Remove $Path"
        return
    }
    try {
        Remove-Item -Force -LiteralPath $Path
    } catch {
        # best-effort
    }
}

if (-not (Test-Path -LiteralPath $TmpDir)) {
    Write-Host "No tmp dir found: $TmpDir"
    exit 0
}

Write-Host "Recovering self-play chunks from: $TmpDir"
Write-Host "- Dataset: $Dataset"
Write-Host "- PGN:     $PgnOut"
Write-Host "- Wins:    $WinsPgnOut"

# Merge JSONL chunks
$jsonl = Get-ChildItem -LiteralPath $TmpDir -Filter "chunk_*.jsonl" -File -ErrorAction SilentlyContinue | Sort-Object Name
if ($null -ne $jsonl -and $jsonl.Count -gt 0) {
    Ensure-ParentDir $Dataset
    if (-not $WhatIf) {
        if (-not (Test-Path -LiteralPath $Dataset)) {
            New-Item -ItemType File -Force -Path $Dataset | Out-Null
        }
    }

    $bytes = 0
    foreach ($f in $jsonl) {
        $bytes += [int64]$f.Length
        Append-File -Source $f.FullName -Dest $Dataset
        Remove-FileSafe $f.FullName
    }
    Write-Host "Merged JSONL chunks: $($jsonl.Count) files ($bytes bytes)"
} else {
    Write-Host "No JSONL chunks found."
}

# Merge PGN chunks
$pgn = Get-ChildItem -LiteralPath $TmpDir -Filter "chunk_*.pgn" -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -notlike "*.wins.pgn" } |
    Sort-Object Name
if ($null -ne $pgn -and $pgn.Count -gt 0) {
    $bytes = 0
    foreach ($f in $pgn) {
        $bytes += [int64]$f.Length
        Append-File -Source $f.FullName -Dest $PgnOut
        Remove-FileSafe $f.FullName
    }
    Write-Host "Merged PGN chunks: $($pgn.Count) files ($bytes bytes)"
} else {
    Write-Host "No PGN chunks found."
}

# Merge wins PGN chunks
$wins = Get-ChildItem -LiteralPath $TmpDir -Filter "chunk_*.wins.pgn" -File -ErrorAction SilentlyContinue | Sort-Object Name
if ($null -ne $wins -and $wins.Count -gt 0) {
    $bytes = 0
    foreach ($f in $wins) {
        $bytes += [int64]$f.Length
        Append-File -Source $f.FullName -Dest $WinsPgnOut
        Remove-FileSafe $f.FullName
    }
    Write-Host "Merged wins PGN chunks: $($wins.Count) files ($bytes bytes)"
} else {
    Write-Host "No wins PGN chunks found."
}

# Clean directory if empty
try {
    # Only remove the directory if there are no remaining chunk artifacts.
    $remainingChunks = Get-ChildItem -LiteralPath $TmpDir -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like 'chunk_*' }
    if ($null -eq $remainingChunks -or $remainingChunks.Count -eq 0) {
        if ($WhatIf) {
            Write-Host "[WhatIf] Remove directory $TmpDir"
        } else {
            Remove-Item -Force -LiteralPath $TmpDir
        }
        Write-Host "Cleaned: $TmpDir"
    } else {
        Write-Host "Remaining chunk files in tmp dir: $($remainingChunks.Count) (not removed)"
    }
} catch {
    # best-effort
}

Write-Host "Done."