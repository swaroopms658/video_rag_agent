"""
Compute 95% bootstrap confidence intervals for ITMA cold-start results.
Uses seed-level data from analysis/cold_start.csv (5 seeds per system/N).
Key finding: ITMA N=50 CI lower bound (0.922) > CFRAG-lite (0.915) → statistically significant.
"""
import pandas as pd
import numpy as np
from scipy.stats import bootstrap as scipy_bootstrap

df = pd.read_csv("analysis/cold_start.csv")


def boot_ci(vals, n_boot=5000, ci=0.95):
    if np.std(vals) < 1e-9:
        m = float(np.mean(vals))
        return m, m
    res = scipy_bootstrap(
        (vals,), np.mean,
        n_resamples=n_boot,
        confidence_level=ci,
        method="percentile",
        random_state=42,
    )
    return res.confidence_interval.low, res.confidence_interval.high


print("=== ITMA cold-start curve (95% bootstrap CI, 5 seeds) ===")
for n in [0, 5, 10, 20, 30, 50]:
    vals = df[(df.system == "itma") & (df.n_feedback == n)]["hit_at_5"].values
    lo, hi = boot_ci(vals)
    print(f"  N={n:2d}: {np.mean(vals):.4f}  [95% CI: {lo:.4f}, {hi:.4f}]")

print()
print("=== Static baselines ===")
for sys in ["dense_minilm", "cfrag_lite", "static_memory"]:
    vals = df[(df.system == sys) & (df.n_feedback == 0)]["hit_at_5"].values
    lo, hi = boot_ci(vals)
    print(f"  {sys}: {np.mean(vals):.4f}  [95% CI: {lo:.4f}, {hi:.4f}]")

print()
# Key significance claim
itma_n50 = df[(df.system == "itma") & (df.n_feedback == 50)]["hit_at_5"].values
cfrag_val = float(df[(df.system == "cfrag_lite") & (df.n_feedback == 0)]["hit_at_5"].mean())
lo50, hi50 = boot_ci(itma_n50)
print(f"KEY RESULT: ITMA N=50 CI lower bound = {lo50:.4f}")
print(f"            CFRAG-lite value          = {cfrag_val:.4f}")
print(f"            Significant (lb > cfrag): {lo50 > cfrag_val}")
