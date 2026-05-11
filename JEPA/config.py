# JEPA/config.py — backward-compatibility shim for pre-migration checkpoints.
#
# Old checkpoints were saved while sys.path included JEPA/, so Python serialized
# Config with __module__ = 'config'.  At load time pickle does:
#     import config; config.Config
# which resolves here.  We re-export the same class object from its new home so
# unpickling succeeds without touching any .pt files.
#
# DO NOT DELETE THIS FILE.

from JEPA.experiments.exp_001_vit_jepa_baseline.config import Config  # noqa: F401

__all__ = ["Config"]
