"""
Shared pytest configuration: put JEPA/ on sys.path so test modules can
import from debug_runner without installing the package.
"""
import sys
from pathlib import Path

# JEPA/ needs to be on sys.path for "from dashboard.debug_runner import ..."
# and for "from encoder import Encoder" etc.
JEPA_DIR = Path(__file__).parent.parent.parent  # Code Repo/JEPA/
sys.path.insert(0, str(JEPA_DIR))
