"""mini_env — pure-numpy 32x32 LS20-Level-1 mini environment.

Plug-compatible with claude_automate / JEPA trainers via the attribute surface
documented on MiniLS20Env. No arcengine dependency.
"""

from mini_env.env import MiniLS20Env
from mini_env.loader import EnvConfig, load_level
from mini_env.state import EnvState

__all__ = ["MiniLS20Env", "EnvState", "EnvConfig", "load_level"]
