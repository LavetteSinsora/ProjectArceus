"""Golden-frame test: locks the level_01 reset render exactly.

Future drift in the renderer fails this test. To re-generate the golden
intentionally, delete the .npy and run pytest once — it will fail with a
clear message; then regenerate via the dev helper in mini_env/README.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mini_env.env import MiniLS20Env

_HERE = Path(__file__).resolve().parent
_CONFIGS = _HERE.parents[0] / "configs"
_GOLDEN = _HERE / "golden" / "level_01_frame.npy"


def test_level_01_render_matches_golden():
    if not _GOLDEN.exists():
        pytest.fail(
            f"golden frame missing at {_GOLDEN}; regenerate via "
            "`uv run python -c \"from mini_env.env import MiniLS20Env; "
            "import numpy as np; "
            "np.save('mini_env/tests/golden/level_01_frame.npy', "
            "MiniLS20Env('mini_env/configs/level_01.json').reset())\"`"
        )
    env = MiniLS20Env(str(_CONFIGS / "level_01.json"))
    frame = env.reset()
    golden = np.load(_GOLDEN)
    assert frame.shape == golden.shape
    assert frame.dtype == golden.dtype
    assert np.array_equal(frame, golden), (
        "Renderer output drifted from the locked level_01 golden frame. "
        "If the change is intentional, regenerate golden/level_01_frame.npy "
        "and review the diff."
    )
