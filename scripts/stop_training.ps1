param(
    # What to stop. Default targets the AZ loop python process.
    [string]$Match = "train\az_loop.py",

    # If set, also try to close VS Code windows after stopping training.
    [switch]$CloseVSCode
)

$ErrorActionPreference = 'Stop'

# Find python processes whose command line contains our training script.
$procs = Get-CimInstance Win32_Process | Where-Object {
    ($_.Name -ieq 'python.exe' -or $_.Name -ieq 'pythonw.exe') -and
    ($_.CommandLine -and $_.CommandLine -like "*$Match*")
}

if (-not $procs) {
    Write-Host "No training python process found (match='$Match')."
} else {
    foreach ($p in $procs) {
        Write-Host "Stopping PID $($p.ProcessId): $($p.CommandLine)"
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
}

if ($CloseVSCode) {
    Get-Process Code -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
}
