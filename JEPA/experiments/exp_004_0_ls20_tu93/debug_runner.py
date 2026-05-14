"""
Debug-episode runner for exp_004_0_ls20_tu93.

exp_004 shares the encoder/state predictor/action predictor architecture with
exp_003_4_no_resampler_self_attn. The only differences at inference time are:

  - action embeddings and policies are per-env dicts in the checkpoint
    (`action_embeds`, `policies`), keyed by short env name ("ls20" / "tu93")
  - the env is chosen by the dashboard from `cfg.env_names`
  - end-of-life detection dispatches by env (LS20 = count_lives; TU93 = is_terminal)

So we delegate the rollout to exp_003_4's runner by building a temporary
exp_003_4-shaped checkpoint dict in a tempfile (extracting the env-specific
policy + action_embed) and patching `cfg.game_id` to the chosen env's full ID.
This keeps the (heavily-instrumented) rollout logic in one place.

Cross-env play is well-defined for exp_004 because both envs were trained on;
no warning banner is emitted.
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


def run_debug_episode(checkpoint_path: str,
                      env_name: str | None = None,
                      max_steps: int = 200) -> dict:
    """
    Args:
      checkpoint_path: path to an exp_004 checkpoint (.pt). Expected keys:
        encoder, state_predictor, action_predictor, action_embeds (dict),
        policies (dict), config (dict), step (int).
      env_name: short env name ("ls20" / "tu93"). Defaults to the first entry
        in the checkpoint's `config.env_names`.
      max_steps: forwarded to exp_003_4's rollout loop.

    Returns the standard per-timestep dict, with:
      experiment = "exp_004_0_ls20_tu93"
      env_name   = <ls20|tu93>
    """
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    cfg_dict = dict(ck["config"]) if isinstance(ck["config"], dict) else ck["config"]
    if isinstance(cfg_dict, dict):
        env_names = list(cfg_dict.get("env_names", ("ls20", "tu93")))
    else:
        env_names = list(getattr(cfg_dict, "env_names", ("ls20", "tu93")))

    if env_name is None:
        env_name = env_names[0]
    if env_name not in env_names:
        raise ValueError(
            f"env_name={env_name!r} not in this experiment's trained envs "
            f"{env_names}. (Cross-env play for exp_004 across these two envs is "
            "valid; an unknown env_name is not.)"
        )

    action_embeds = ck.get("action_embeds") or {}
    policies      = ck.get("policies")      or {}
    if env_name not in action_embeds or env_name not in policies:
        raise KeyError(
            f"Checkpoint is missing action_embeds/policies for env {env_name!r}. "
            f"Available action_embeds: {list(action_embeds)}; "
            f"policies: {list(policies)}."
        )

    # Build an exp_003_4-shaped checkpoint dict pinned to the chosen env.
    # Patch cfg.game_id to the env-specific full game id so exp_003_4's runner
    # (which calls arc.make(cfg.game_id) and make_env) ends up in the right env.
    full_gid = full_game_id(env_name)
    if isinstance(cfg_dict, dict):
        cfg_dict = dict(cfg_dict)
        cfg_dict["game_id"] = full_gid
        # exp_003_4 Config doesn't know about env_names / game_ids /
        # buffer_size_per_env / etc.; its debug_runner filters unknown fields
        # via dataclasses.fields(Config), so leaving them in is harmless.
    proxy_ck = {
        "encoder":          ck["encoder"],
        "state_predictor":  ck["state_predictor"],
        "action_predictor": ck["action_predictor"],
        "action_embed":     action_embeds[env_name],
        "policy":           policies[env_name],
        "config":           cfg_dict,
        "step":             ck.get("step", 0),
    }

    # Write the proxy checkpoint to a tempfile and delegate to exp_003_4's
    # debug runner. We pass env_name=env_name explicitly so its make_env path
    # respects our choice (even though cfg.game_id already encodes it).
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

    # Relabel for the dashboard panel router.
    data["experiment"] = "exp_004_0_ls20_tu93"
    data["env_name"]   = env_name
    # exp_004 was trained on this env — drop any cross-env warning that the
    # upstream runner might have produced (shouldn't happen because we pinned
    # cfg.game_id, but be defensive).
    data.pop("warning", None)
    return data


__all__ = ["run_debug_episode"]


if __name__ == "__main__":
    import json
    ckpt = sys.argv[1] if len(sys.argv) > 1 else (
        "JEPA/experiments/exp_004_0_ls20_tu93/checkpoints/step_500000_final.pt"
    )
    env  = sys.argv[2] if len(sys.argv) > 2 else "ls20"
    out = run_debug_episode(ckpt, env_name=env, max_steps=5)
    print(f"experiment={out['experiment']}  env={out['env_name']}  "
          f"steps={out['episode_steps']}  truncated={out['truncated']}")
    print(f"JSON size: {len(json.dumps(out)) // 1024} KB")
