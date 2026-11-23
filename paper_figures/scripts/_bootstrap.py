"""Allow every script to run directly in PyCharm or from any working directory."""
import os
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(BASE / '.cache/matplotlib'))
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))
