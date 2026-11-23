"""Compatibility entry point for the selected research results."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[1] / 'paper_figures/build_all.py'), run_name='__main__')
