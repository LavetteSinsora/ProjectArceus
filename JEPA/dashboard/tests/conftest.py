"""
Shared pytest configuration: put repo root on sys.path so the JEPA package
(and its sub-packages: JEPA.shared, JEPA.experiments, JEPA.dashboard) are
importable without a package install.
"""
import sys
from pathlib import Path

# Code Repo/ needs to be on sys.path so "import JEPA" resolves.
REPO_ROOT = Path(__file__).parent.parent.parent.parent  # tests → dashboard → JEPA → Code Repo
sys.path.insert(0, str(REPO_ROOT))
