$python = "D:/NullMove/.venv/Scripts/python.exe"

# Sends a tiny UCI session to the engine.
@(
  "uci",
  "isready",
  "ucinewgame",
  "position startpos moves e2e4 e7e5",
  "go depth 4",
  "quit"
) | & $python -m nullmove.uci
