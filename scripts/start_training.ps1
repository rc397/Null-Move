param(
    [int]$Iters = 1000000,
    [int]$GamesPerIter = 100,
    [int]$Sims = 200,
    [int]$BatchSize = 256,
    [int]$MaxPlies = 250,
    [int]$TrainEpochs = 1,
    [int]$TrainBatch = 256,
    [double]$Lr = 1e-3,
    # Bigger replay buffer => more RAM usage and usually smoother learning.
    [int]$ReplayMax = 2000000,

    # Larger top-k policy packing uses more RAM per position (and a bit more compute).
    [int]$PolicyTopK = 128,

    # Progress tracking (optional eval; new weights vs previous weights).
    # These games are extra compute, but they give you a real strength trend signal.
    [int]$EvalGames = 20,
    [int]$EvalEvery = 1,
    [int]$EvalEveryGames = 50,
    [int]$EvalSims = 500,
    [int]$EvalBatchSize = 32,
    [int]$EvalOpeningPlies = 6,

    # Slightly discourage draws in training targets.
    [double]$DrawValue = -0.2,

    # Mildly discourage repeating recent positions (keeps some shuffling possible).
    [double]$RepeatPenalty = 0.10,
    [int]$RepeatWindow = 12,

    # Run self-play in multiple processes to use more CPU (and increase GPU inference throughput).
    [int]$SelfplayWorkers = 6,

    # ResNet size ("neurons").
    [int]$NetChannels = 96,
    [int]$NetBlocks = 8,
    [switch]$NoNetAmp,

    # Save checkpoint PGNs into Self_Play_Games.
    # Note: with multi-process self-play, PGNs are merged safely after workers finish.
    [int]$PgnEvery = 100,

    # Exploration knobs (to avoid repeating the same games)
    [int]$TempPlies = 30,
    [double]$TempInit = 1.25,
    [double]$TempFinal = 0.50,
    [double]$SamplePAfterTemp = 0.15,
    [double]$DirichletAlpha = 0.30,
    [double]$DirichletEps = 0.25,

    # If set, clears the MCTS tree every move (more variety, slower/weaker)
    [switch]$ClearTreeEveryMove,

    # Optional: also open VS Code on launch if `code` is available.
    [switch]$OpenVSCode,

    # If set, do not tail the log (run fully detached).
    [switch]$NoTail,

    # Approx CPU share by limiting threads/affinity.
    # keep to 99% as close to 100% as possible
    [double]$CpuShare = 0.99
)

$ErrorActionPreference = 'Stop'

$Root = "D:\NullMove"
$Py = Join-Path $Root ".venv\Scripts\python.exe"
$TrainScript = Join-Path $Root "train\az_loop.py"

$SelfPlayDir = Join-Path $Root "Self_Play_Games"
New-Item -ItemType Directory -Force -Path $SelfPlayDir | Out-Null

$Dataset = Join-Path $Root "az_dataset.jsonl"
$Weights = Join-Path $Root "az.pt"
$PgnOut = Join-Path $SelfPlayDir "az_progress.pgn"

$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogOutPath = Join-Path $LogDir "az_loop_$Stamp.out.log"
$LogErrPath = Join-Path $LogDir "az_loop_$Stamp.err.log"

# Pre-create log files so hardlink creation doesn't race cmd.exe redirection.
New-Item -ItemType File -Force -Path $LogOutPath | Out-Null
New-Item -ItemType File -Force -Path $LogErrPath | Out-Null

# Stable paths you can always tail (used by boot-start tasks too).
$LatestOut = Join-Path $LogDir "az_loop_latest.out.log"
$LatestErr = Join-Path $LogDir "az_loop_latest.err.log"
$LatestInfo = Join-Path $LogDir "az_loop_latest_paths.txt"

# Live per-game move log (safe across multi-process self-play).
$SelfplayLiveLog = Join-Path $LogDir "selfplay_live.log"
New-Item -ItemType File -Force -Path $SelfplayLiveLog | Out-Null

if (-not (Test-Path $Py)) {
    throw "Python venv not found at $Py"
}
if (-not (Test-Path $TrainScript)) {
    throw "Training script not found at $TrainScript"
}

# Safety: don't start multiple concurrent loops.
$existing = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'NullMove\\train\\az_loop\.py' }
if ($null -ne $existing) {
    $pids = ($existing | Select-Object -ExpandProperty ProcessId) -join ', '
    throw "Training loop already running (pid(s): $pids). Stop it before starting a new one."
}

# Try to bias CPU usage to ~$CpuShare while avoiding oversubscription.
# With multi-process self-play, each worker inherits these env vars, so we scale threads per-process.
$logical = [Environment]::ProcessorCount
$workersEff = [Math]::Max(1, [int]$SelfplayWorkers)
$threadsTotal = [Math]::Max(1, [int][Math]::Floor($logical * $CpuShare))
$threads = [Math]::Max(1, [int][Math]::Floor($threadsTotal / $workersEff))
$env:OMP_NUM_THREADS = "$threads"
$env:MKL_NUM_THREADS = "$threads"
$env:NUMEXPR_NUM_THREADS = "$threads"
$env:TORCH_NUM_THREADS = "$threads"

$mask = 0
for ($i = 0; $i -lt $threadsTotal; $i++) {
    $mask = $mask -bor (1 -shl $i)
}

if ($OpenVSCode) {
    $code = Get-Command code -ErrorAction SilentlyContinue
    if ($null -ne $code) {
        Start-Process -FilePath $code.Source -ArgumentList @($Root) -WorkingDirectory $Root | Out-Null
    }
}

$args = @(
    "-u",
    $TrainScript,
    "--iters", "$Iters",
    "--games-per-iter", "$GamesPerIter",
    "--sims", "$Sims",
    "--batch-size", "$BatchSize",
    "--selfplay-workers", "$SelfplayWorkers",
    "--max-plies", "$MaxPlies",
    "--train-epochs", "$TrainEpochs",
    "--train-batch", "$TrainBatch",
    "--lr", "$Lr",
    "--replay-max", "$ReplayMax",
    "--policy-topk", "$PolicyTopK",
    "--dataset", $Dataset,
    "--weights", $Weights,
    "--net-channels", "$NetChannels",
    "--net-blocks", "$NetBlocks",
    "--pgn-out", $PgnOut,
    "--pgn-every", "$PgnEvery",
    "--eval-games", "$EvalGames",
    "--eval-every", "$EvalEvery",
    "--eval-every-games", "$EvalEveryGames",
    "--eval-sims", "$EvalSims",
    "--eval-batch-size", "$EvalBatchSize",
    "--print-moves"
    ,"--eval-opening-plies", "$EvalOpeningPlies"
)

if ($NoNetAmp) {
    $args += "--no-net-amp"
} else {
    $args += "--net-amp"
}

$args += @(
    "--temp-plies", "$TempPlies",
    "--temp-init", "$TempInit",
    "--temp-final", "$TempFinal",
    "--sample-p-after-temp", "$SamplePAfterTemp",
    "--dirichlet-alpha", "$DirichletAlpha",
    "--dirichlet-eps", "$DirichletEps"
)

$args += @(
    "--draw-value", "$DrawValue"
)

$args += @(
    "--repeat-penalty", "$RepeatPenalty",
    "--repeat-window", "$RepeatWindow"
)

$args += @(
    "--selfplay-live-log", $SelfplayLiveLog
)

if ($ClearTreeEveryMove) {
    $args += "--clear-tree-every-move"
}

Write-Host "Starting NullMove AZ training..." 
Write-Host "- Dataset:  $Dataset"
Write-Host "- Weights:  $Weights"
Write-Host "- PGN:      $PgnOut (every $PgnEvery games)"
Write-Host "- MaxPlies: $MaxPlies (per self-play game cap)"
Write-Host "- CPU:      $threads/$logical threads per process (workers=$SelfplayWorkers, target~$([Math]::Round(100*$CpuShare))%)"
Write-Host "- Log(out): $LogOutPath"
Write-Host "- Log(err): $LogErrPath"

$argString = ($args | ForEach-Object {
    if ($_ -match '\s') { '"' + $_ + '"' } else { $_ }
}) -join ' '

# IMPORTANT (Windows): PowerShell's Start-Process stream redirection can break Python multiprocessing
# (workers fail to spawn / parent appears stuck). Use cmd.exe redirection so handles are inherited.
# cmd.exe needs the *entire* command wrapped in outer quotes when the executable path is quoted.
$cmdLineInner = "`"$Py`" $argString 1>`"$LogOutPath`" 2>`"$LogErrPath`""
$cmdLine = "`"$cmdLineInner`""
$p = Start-Process -FilePath $env:ComSpec -ArgumentList @('/c', $cmdLine) -WorkingDirectory $Root -PassThru

# Try to find the actual python PID we just launched (cmd.exe is just a wrapper).
Start-Sleep -Milliseconds 400
$pyProc = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'NullMove\\train\\az_loop\.py' } |
    Sort-Object CreationDate -Descending |
    Select-Object -First 1

$launchedPid = if ($null -ne $pyProc) { [int]$pyProc.ProcessId } else { [int]$p.Id }

if ($null -ne $pyProc) {
    try {
        $pp = Get-Process -Id $launchedPid -ErrorAction Stop
        try { $pp.PriorityClass = 'High' } catch { }
        # Don't set affinity when using multiple worker processes; it would pin ALL workers to the same cores.
        if ($workersEff -le 1) {
            try { $pp.ProcessorAffinity = [IntPtr]$mask } catch { }
        }
    } catch { }
}

Write-Host "Started PID $launchedPid. Tail the log with:" 
Write-Host "  Get-Content -Path $LatestOut -Wait"
Write-Host "  Get-Content -Path $LatestErr -Wait"
Write-Host "  Get-Content -Path $SelfplayLiveLog -Wait   # live self-play moves"
Write-Host "  .\\scripts\\tail_selfplay_workers.ps1 -Workers $SelfplayWorkers   # open one tail window per worker"

# Update stable 'latest' log links so you don't need to chase timestamps.
try {
    if (Test-Path $LatestOut) { Remove-Item -Force $LatestOut -ErrorAction SilentlyContinue }
    if (Test-Path $LatestErr) { Remove-Item -Force $LatestErr -ErrorAction SilentlyContinue }

    # Hardlinks do not require admin privileges; they point to the same file contents.
    # The redirected files should already exist after Start-Process returns.
    New-Item -ItemType HardLink -Path $LatestOut -Target $LogOutPath -ErrorAction Stop | Out-Null
    New-Item -ItemType HardLink -Path $LatestErr -Target $LogErrPath -ErrorAction Stop | Out-Null

    @(
        "Started: $Stamp",
        "PID: $launchedPid",
        "OUT: $LogOutPath",
        "ERR: $LogErrPath",
        "LATEST_OUT: $LatestOut",
        "LATEST_ERR: $LatestErr"
    ) | Set-Content -Path $LatestInfo -Encoding utf8
} catch {
    Write-Warning "Could not create latest log links: $($_.Exception.Message)"
    Write-Host "Fallback tail commands:" 
    Write-Host "  Get-Content -Path $LogOutPath -Wait"
    Write-Host "  Get-Content -Path $LogErrPath -Wait"
}

# Keep the script running if launched interactively (unless suppressed).
if ((-not $NoTail) -and ($Host.Name -ne 'ServerRemoteHost')) {
    # Always tail the stable latest path so it matches your boot-start setup.
    if (Test-Path $LatestOut) {
        Get-Content -Path $LatestOut -Wait
    } else {
        Get-Content -Path $LogOutPath -Wait
    }
}
