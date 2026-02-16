"""Pytest configuration for jaato-tui tests.

Run tests with: PYTHONPATH=jaato-tui pytest jaato-tui/tests/ --import-mode=importlib
"""

import sys
from pathlib import Path

# Add jaato-tui directory to path for imports
rich_client_dir = Path(__file__).parent.parent
if str(rich_client_dir) not in sys.path:
    sys.path.insert(0, str(rich_client_dir))
