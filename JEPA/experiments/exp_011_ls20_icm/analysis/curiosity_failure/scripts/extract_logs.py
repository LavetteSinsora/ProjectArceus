"""Extract per-update log trajectories (all seeds) for L1 and L2 into log_traj.json."""
import json, glob, os
ROOT = "/Users/chrishe/Desktop/同步空间/Chris/Chris/Obsidian Vault/UCSD/Course Notes/CSE 190 Deep RL/Group Project/Code Repo"
EXP = {
    "L1": "JEPA/experiments/exp_011_ls20_icm/exp_011_0_icm_baseline/runs/*",
    "L2": "JEPA/experiments/exp_011_ls20_icm/exp_011_2_icm_ls20_l2/runs/*",
}
FIELDS = ["step", "forward_error_mean", "forward_loss", "intrinsic_reward_mean",
          "intrinsic_reward_std", "inverse_acc", "policy_entropy", "feat_effective_rank"]
out = {}
for lab, pat in EXP.items():
    out[lab] = {}
    for d in sorted(glob.glob(os.path.join(ROOT, pat))):
        seed = "seed" + d.split("seed")[1][0]
        rows = [json.loads(l) for l in open(os.path.join(d, "metrics.jsonl"))]
        series = {f: [] for f in FIELDS}
        fr = None
        for r in rows:
            for f in FIELDS:
                series[f].append(r.get(f))
            if r.get("first_reward_step") and fr is None:
                fr = r["first_reward_step"]
        # final eval success
        evals = [r for r in rows if r.get("success_rate") is not None]
        best = max((e["success_rate"] for e in evals), default=0.0)
        series["first_reward_step"] = fr
        series["best_eval_succ"] = best
        out[lab][seed] = series
with open(os.path.join(ROOT, "JEPA/experiments/exp_011_ls20_icm/analysis/curiosity_failure/log_traj.json"), "w") as f:
    json.dump(out, f)
print("wrote log_traj.json")
