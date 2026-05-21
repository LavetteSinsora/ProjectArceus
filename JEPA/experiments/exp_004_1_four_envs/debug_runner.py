"""
Debug-episode runner for exp_004_1_four_envs.

exp_004_1 shares the encoder/state predictor/action predictor architecture with
exp_003_4_no_resampler_self_attn. The differences at inference time are:

  - action embeddings and policies are per-env dicts in the checkpoint
    (`action_embeds`, `policies`), keyed by short env name
    ("ls20" / "tu93" / "re86" / "g50t")
  - the env is chosen by the dashboard from `cfg.env_names`
  - end-of-life dispatches by env (LS20 = count_lives; TU93/RE86/G50T = is_terminal
    unless a probe-driven detector replaces the default)

We delegate the rollout to exp_003_4's runner by building a temporary
exp_003_4-shaped checkpoint dict (extracting the env-specific policy and
action_embed) and patching `cfg.game_id` to the chosen env's full ID. This
keeps the (heavily-instrumented) rollout logic in one place.

Cross-env play is well-defined for exp_004_1 because all four envs were trained
on; no warning banner is emitted.
"""

from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path

import torch

_repo_root = Path(__file__).parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from JEPA.shared.env_wrapper import full_game_id


DEFAULT_ENV_NAMES = ("ls20", "tu93", "re86", "g50t")


def run_debug_episode(checkpoint_path: str,
                      env_name: str | None = None,
                      max_steps: int = 200) -> dict:
    """
    Args:
      checkpoint_path: path to an exp_004_1 checkpoint (.pt). Expected keys:
        encoder, state_predictor, action_predictor, action_embeds (dict),
        policies (dict), config (dict), step (int).
      env_name: short env name ("ls20" / "tu93" / "re86" / "g50t"). Defaults to
        the first entry in the checkpoint's `config.env_names`.
      max_steps: forwarded to exp_003_4's rollout loop.

    Returns the standard per-timestep dict, with:
      experiment = "exp_004_1_four_envs"
      env_name   = <ls20|tu93|re86|g50t>
    """
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    cfg_dict = dict(ck["config"]) if isinstance(ck["config"], dict) else ck["config"]
    if isinstance(cfg_dict, dict):
        env_names = list(cfg_dict.get("env_names", DEFAULT_ENV_NAMES))
    else:
        env_names = list(getattr(cfg_dict, "env_names", DEFAULT_ENV_NAMES))

    if env_name is None:
        env_name = env_names[0]
    if env_name not in env_names:
        raise ValueError(
            f"env_name={env_name!r} not in this experiment's trained envs "
            f"{env_names}. (Cross-env play for exp_004_1 across these four envs "
            "is valid; an unknown env_name is not.)"
        )

    action_embeds = ck.get("action_embeds") or {}
    policies      = ck.get("policies")      or {}
    if env_name not in action_embeds or env_name not in policies:
        raise KeyError(
            f"Checkpoint is missing action_embeds/policies for env {env_name!r}. "
            f"Available action_embeds: {list(action_embeds)}; "
            f"policies: {list(policies)}."
        )

    full_gid = full_game_id(env_name)
    if isinstance(cfg_dict, dict):
        cfg_dict = dict(cfg_dict)
        cfg_dict["game_id"] = full_gid

    proxy_ck = {
        "encoder":          ck["encoder"],
        "state_predictor":  ck["state_predictor"],
        "action_predictor": ck["action_predictor"],
        "action_embed":     action_embeds[env_name],
        "policy":           policies[env_name],
        "config":           cfg_dict,
        "step":             ck.get("step", 0),
    }

    upstream = importlib.import_module(
        "JEPA.experiments.exp_003_4_no_resampler_self_attn.debug_runner"
    )

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tf:
        tmp_path = Path(tf.name)
    try:
        torch.save(proxy_ck, tmp_path)
        data = upstream.run_debug_episode(
            str(tmp_path), env_name=env_name, max_steps=max_steps,
        )
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass

    data["experiment"] = "exp_004_1_four_envs"
    data["env_name"]   = env_name
    data.pop("warning", None)
    return data


__all__ = ["run_debug_episode"]


if __name__ == "__main__":
    import json
    ckpt = sys.argv[1] if len(sys.argv) > 1 else (
        "JEPA/experiments/exp_004_1_four_envs/checkpoints/step_500000_final.pt"
    )
    env  = sys.argv[2] if len(sys.argv) > 2 else "ls20"
    out = run_debug_episode(ckpt, env_name=env, max_steps=5)
    print(f"experiment={out['experiment']}  env={out['env_name']}  "
          f"steps={out['episode_steps']}  truncated={out['truncated']}")
    print(f"JSON size: {len(json.dumps(out)) // 1024} KB")
