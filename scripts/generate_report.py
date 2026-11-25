"""Generate reproducible Markdown and CSV reports from existing experiment results.

Usage:
    python scripts/generate_report.py --output-dir reports
    python scripts/generate_report.py --format csv --model-types loop
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from experiment_analysis.cli import report_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(report_main())
