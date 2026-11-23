"""Run with PyCharm or Python; a single model instance, loopback only."""
import sys
from pathlib import Path
import uvicorn

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    uvicorn.run("main:app", host="127.0.0.1", port=8000, workers=1, reload=False)
