$python = "D:/NullMove/.venv/Scripts/python.exe"

$out = "D:/NullMove/Self_Play_Games/smoke_selfplay.pgn"
& $python scripts/selfplay.py --games 4 --depth 6 --out $out
Write-Host "Wrote $out"
