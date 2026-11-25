"""Compare dense and loop models under matched-parameter or matched-compute conditions.

Usage:
    python scripts/compare_models.py --category matched-parameter
    python scripts/compare_models.py --metric clean_loss --format csv
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiment_analysis.cli import compare_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(compare_main())
