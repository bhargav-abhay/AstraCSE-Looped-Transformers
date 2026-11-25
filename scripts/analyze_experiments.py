"""Discover, validate and summarise existing AstraCSE experiment results.

Usage:
    python scripts/analyze_experiments.py --experiments-dir paper/data/results
    python scripts/analyze_experiments.py --format json --output-dir reports
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiment_analysis.cli import analysis_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(analysis_main())
