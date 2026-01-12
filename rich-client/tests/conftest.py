"""Pytest configuration for rich-client tests.

Run tests with: PYTHONPATH=rich-client pytest rich-client/tests/ --import-mode=importlib
"""

import sys
from pathlib import Path

# Add rich-client directory to path for imports
rich_client_dir = Path(__file__).parent.parent
if str(rich_client_dir) not in sys.path:
    sys.path.insert(0, str(rich_client_dir))
