"""Compute breakeven-N: at what N does ITMA first match each baseline's H@5?

Usage:
    python scripts/compute_breakeven_n.py
    python scripts/compute_breakeven_n.py --csv analysis/results/cold_start_n89.csv
"""

import argparse
import pandas as pd


def compute(csv_path: str):
    df = pd.read_csv(csv_path)
    itma = df[df.system == "itma"].groupby("n_feedback")["hit_at_5"].mean()

    baselines = [s for s in df["system"].unique() if s != "itma"]
    print(f"\nBreakeven-N  ({csv_path})\n{'-'*45}")
    for name in baselines:
        threshold = df[df.system == name]["hit_at_5"].mean()
        crossed = itma[itma >= threshold]
        n = int(crossed.index[0]) if len(crossed) else "never"
        print(f"  ITMA matches {name:<18} (H@5={threshold:.4f}) at N={n}")

    print("\nITMA cold-start curve (mean across seeds):")
    for n, v in itma.items():
        print(f"  N={n:>3}: {v:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="analysis/cold_start.csv")
    args = parser.parse_args()
    compute(args.csv)


if __name__ == "__main__":
    main()
