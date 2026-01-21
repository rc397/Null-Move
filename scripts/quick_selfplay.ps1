$python = "D:/NullMove/.venv/Scripts/python.exe"

& $python scripts/selfplay.py --games 4 --depth 6 --out selfplay.pgn
Write-Host "Wrote selfplay.pgn"
