param(
    # What to stop. Default targets the AZ loop python process.
    [string]$Match = "train\az_loop.py",

    # If set, also try to close VS Code windows after stopping training.
    [switch]$CloseVSCode,

    # Max seconds to wait for graceful shutdown before force-killing.
    [int]$GracefulTimeout = 300,

    # If set, skip graceful shutdown and force-kill immediately.
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

$Root = Split-Path -Parent $PSScriptRoot
$StopFile = Join-Path $Root ".az_stop"

# Find python processes whose command line contains our training script.
$procs = Get-CimInstance Win32_Process | Where-Object {
    ($_.Name -ieq 'python.exe' -or $_.Name -ieq 'pythonw.exe') -and
    ($_.CommandLine -and $_.CommandLine -like "*$Match*")
}

if (-not $procs) {
    Write-Host "No training python process found (match='$Match')."
    # Clean up any stale stop file
    if (Test-Path $StopFile) { Remove-Item $StopFile -Force }
    exit 0
}

if ($Force) {
    # Force kill immediately
    foreach ($p in $procs) {
        Write-Host "Force-stopping PID $($p.ProcessId): $($p.CommandLine)"
        try {
            Stop-Process -Id $p.ProcessId -Force
        } catch {
            try {
                $null = Invoke-CimMethod -InputObject $p -MethodName Terminate
            } catch {
                Write-Host "Failed to terminate PID $($p.ProcessId)"
            }
        }
    }
} else {
    # Graceful shutdown: create stop file and wait
    Write-Host "Requesting graceful shutdown (will wait up to $GracefulTimeout seconds)..."
    Write-Host "Creating stop file: $StopFile"
    
    # Create the stop file to signal shutdown
    "shutdown requested at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $StopFile -Encoding utf8
    
    $pids = $procs | ForEach-Object { $_.ProcessId }
    $startTime = Get-Date
    $allExited = $false
    
    while (-not $allExited -and ((Get-Date) - $startTime).TotalSeconds -lt $GracefulTimeout) {
        $allExited = $true
        foreach ($pid in $pids) {
            try {
                $proc = Get-Process -Id $pid -ErrorAction Stop
                $allExited = $false
                Write-Host "Waiting for PID $pid to finish current iteration... ($(([int]((Get-Date) - $startTime).TotalSeconds))s elapsed)"
            } catch {
                # Process exited
            }
        }
        if (-not $allExited) {
            Start-Sleep -Seconds 5
        }
    }
    
    if ($allExited) {
        Write-Host "Training processes exited gracefully."
    } else {
        Write-Host "Timeout reached. Force-killing remaining processes..."
        foreach ($pid in $pids) {
            try {
                Stop-Process -Id $pid -Force -ErrorAction Stop
                Write-Host "Force-killed PID $pid"
            } catch {
                # Already exited
            }
        }
    }
    
    # Clean up stop file
    if (Test-Path $StopFile) { Remove-Item $StopFile -Force }
}

if ($CloseVSCode) {
    Get-Process Code -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}
