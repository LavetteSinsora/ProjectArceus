"""Dreamer V3 neural-network modules."""

from .world_model import WorldModel
from .actor import Actor
from .heads import ValueHead
from .ema import CriticEMA


def load_models(cfg, device):
    """Build all DV3 nets and move them to `device`.

    Returns:
        wm:         WorldModel (encoder + decoder + RSSM + reward/continue heads + P2E ensemble)
        actor:      task actor π_t
        critic:     task critic v_ψ
        critic_ema: EMA copy of critic (decay 0.98)
        actor_p2e:  exploration actor π_e (None if not used)
        critic_p2e: exploration critic    (None if not used)
        critic_p2e_ema: EMA of exploration critic (None if not used)
    """
    wm = WorldModel(cfg).to(device)
    actor = Actor(cfg).to(device)
    critic = ValueHead(cfg).to(device)
    critic_ema = CriticEMA(critic, decay=cfg.critic_ema_decay)

    actor_p2e = None
    critic_p2e = None
    critic_p2e_ema = None
    if cfg.use_p2e_actor:
        actor_p2e = Actor(cfg).to(device)
        critic_p2e = ValueHead(cfg).to(device)
        critic_p2e_ema = CriticEMA(critic_p2e, decay=cfg.critic_ema_decay)

    return wm, actor, critic, critic_ema, actor_p2e, critic_p2e, critic_p2e_ema
