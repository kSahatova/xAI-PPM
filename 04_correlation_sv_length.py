"""
Cross-strategy comparison table.

Loads segment-level SHAP value results from all four segmentation strategies,
computes per-strategy statistics (length distribution, Spearman, rmcorr),
runs a Friedman test across strategies, and outputs a summary table.

Expected folder structure:
    <OUTPUT_ROOT>/shap_values/bpi17_transition/segment_sv_results.pkl
    <OUTPUT_ROOT>/shap_values/bpi17_random/segment_sv_results.pkl
    <OUTPUT_ROOT>/shap_values/bpi17_per_event/segment_sv_results.pkl
    <OUTPUT_ROOT>/shap_values/bpi17_two_event/segment_sv_results.pkl

Usage:
    python build_comparison_table.py
"""

import os
import os.path as osp
import pickle

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ── Configuration ───────────────────────────────────────────────────
OUTPUT_ROOT = r"D:\PycharmProjects\xAI-PPM\outputs"
SV_DIR = osp.join(OUTPUT_ROOT, "shap_values", "bpi17")
TABLE_DIR = osp.join(OUTPUT_ROOT, "shap_values","comparison")

SAMPLE_NAMES = ["tp", "tn", "fp", "fn"]

STRATEGIES = {
    "transition": osp.join(SV_DIR, "transition", "segment_sv_results.pkl"),
    "random":     osp.join(SV_DIR, "random", "segment_sv_results.pkl"),
    "per_event":  osp.join(SV_DIR, "per_event", "segment_sv_results.pkl"),
    "two_event":  osp.join(SV_DIR, "two_event", "segment_sv_results.pkl"),
}

# Minimum unique lengths required to compute correlation
MIN_UNIQUE_LENGTHS = 3


# ── 1. Load and flatten results ─────────────────────────────────────

def load_and_flatten(pkl_path: str, strategy: str) -> pd.DataFrame:
    """Load a pickle and flatten into one row per segment."""
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)

    sv_results = payload["sv_results"]
    rows = []

    for name in SAMPLE_NAMES:
        for case_idx, res in enumerate(sv_results[name]):
            sv_arr = np.asarray(res["segment_sv"]).ravel()
            seg_ids = res["segment_ids"]

            for seg_idx, (sv_val, seg) in enumerate(zip(sv_arr, seg_ids)):
                rows.append({
                    "strategy": strategy,
                    "sample": name,
                    "case_idx": case_idx,
                    # Unique case key across samples
                    "case_key": f"{name}_{case_idx}",
                    "seg_idx": seg_idx,
                    "seg_length": len(seg),
                    "sv": float(sv_val),
                    "abs_sv": abs(float(sv_val)),
                })

    return pd.DataFrame(rows)


def load_all_strategies() -> pd.DataFrame:
    """Load all strategy results into a single DataFrame."""
    frames = []
    for strategy, path in STRATEGIES.items():
        if not osp.exists(path):
            print(f"  WARNING: {path} not found, skipping '{strategy}'")
            continue
        print(f"  Loading {strategy} from {path}")
        frames.append(load_and_flatten(path, strategy))

    return pd.concat(frames, ignore_index=True)


# ── 2. Per-strategy statistics ──────────────────────────────────────

def compute_length_stats(df: pd.DataFrame) -> dict:
    """Mean ± std of segment length."""
    return {
        "seg_len_mean": df["seg_length"].mean(),
        "seg_len_std": df["seg_length"].std(),
        "n_unique_lengths": df["seg_length"].nunique(),
    }


def compute_global_spearman(df: pd.DataFrame) -> dict:
    """Global Spearman between seg_length and |SV|."""
    n_unique = df["seg_length"].nunique()
    if n_unique < MIN_UNIQUE_LENGTHS:
        return {"spearman_rho": np.nan, "spearman_p": np.nan}

    rho, p = sp_stats.spearmanr(df["seg_length"], df["abs_sv"])
    return {"spearman_rho": rho, "spearman_p": p}


def compute_rmcorr(df: pd.DataFrame) -> dict:
    """Repeated-measures correlation (seg_length vs |SV|, grouped by case).

    Falls back to manual ANCOVA computation if pingouin is not installed.
    """
    n_unique = df["seg_length"].nunique()
    if n_unique < MIN_UNIQUE_LENGTHS:
        return {"rmcorr_r": np.nan, "rmcorr_p": np.nan, "rmcorr_note": "n/a (constant length)"}

    # Need at least 2 observations per case for some cases
    valid_cases = df.groupby("case_key").filter(lambda g: len(g) >= 2)
    if len(valid_cases) < 10:
        return {"rmcorr_r": np.nan, "rmcorr_p": np.nan, "rmcorr_note": "n/a (too few data)"}

    try:
        import pingouin as pg
        result = pg.rm_corr(data=valid_cases, x="seg_length", y="abs_sv", subject="case_key")
        return {
            "rmcorr_r": result["r"].values[0],
            "rmcorr_p": result["pval"].values[0],
            "rmcorr_note": "",
        }
    except ImportError:
        # Manual ANCOVA-based rmcorr
        return _manual_rmcorr(valid_cases)
    except Exception as e:
        return {"rmcorr_r": np.nan, "rmcorr_p": np.nan, "rmcorr_note": f"error: {e}"}


def _manual_rmcorr(df: pd.DataFrame) -> dict:
    """Compute rmcorr via ANCOVA residuals when pingouin is unavailable."""
    from scipy.stats import pearsonr

    # Mean-center both variables within each case
    df = df.copy()
    group_means_x = df.groupby("case_key")["seg_length"].transform("mean")
    group_means_y = df.groupby("case_key")["abs_sv"].transform("mean")
    df["x_centered"] = df["seg_length"] - group_means_x
    df["y_centered"] = df["abs_sv"] - group_means_y

    r, p = pearsonr(df["x_centered"], df["y_centered"])
    return {"rmcorr_r": r, "rmcorr_p": p, "rmcorr_note": "(manual)"}


def compute_case_level_summary(df: pd.DataFrame) -> pd.DataFrame:
    """For each case: max |SV|, sum |SV|, mean |SV|, n_segments."""
    return (
        df.groupby(["strategy", "case_key"])
        .agg(
            max_abs_sv=("abs_sv", "max"),
            sum_abs_sv=("abs_sv", "sum"),
            mean_abs_sv=("abs_sv", "mean"),
            n_segments=("seg_idx", "count"),
        )
        .reset_index()
    )


# ── 3. Friedman test ────────────────────────────────────────────────

def run_friedman_test(case_summary: pd.DataFrame, metric: str = "max_abs_sv") -> dict:
    """Friedman test across strategies for a given metric.

    Only includes cases present in ALL strategies.
    """
    pivot = case_summary.pivot(index="case_key", columns="strategy", values=metric)
    pivot = pivot.dropna()  # keep only cases with all strategies

    if pivot.shape[1] < 3:
        return {"friedman_stat": np.nan, "friedman_p": np.nan,
                "n_cases": pivot.shape[0], "metric": metric,
                "note": "need >= 3 strategies"}

    stat, p = sp_stats.friedmanchisquare(*[pivot[col] for col in pivot.columns])
    return {
        "friedman_stat": stat,
        "friedman_p": p,
        "n_cases": pivot.shape[0],
        "metric": metric,
    }


def run_pairwise_wilcoxon(case_summary: pd.DataFrame, metric: str = "max_abs_sv") -> pd.DataFrame:
    """Pairwise Wilcoxon signed-rank tests with Bonferroni correction."""
    pivot = case_summary.pivot(index="case_key", columns="strategy", values=metric).dropna()
    strategies = list(pivot.columns)
    n_comparisons = len(strategies) * (len(strategies) - 1) // 2

    rows = []
    for i in range(len(strategies)):
        for j in range(i + 1, len(strategies)):
            a, b = strategies[i], strategies[j]
            try:
                stat, p = sp_stats.wilcoxon(pivot[a], pivot[b])
                rows.append({
                    "pair": f"{a} vs {b}",
                    "W": stat,
                    "p_raw": p,
                    "p_bonferroni": min(p * n_comparisons, 1.0),
                    "n": len(pivot),
                })
            except Exception as e:
                rows.append({"pair": f"{a} vs {b}", "W": np.nan,
                             "p_raw": np.nan, "p_bonferroni": np.nan, "note": str(e)})

    return pd.DataFrame(rows)


# ── 4. Build the summary table ──────────────────────────────────────

def build_summary_table(df: pd.DataFrame, case_summary: pd.DataFrame) -> pd.DataFrame:
    """One row per strategy with all metrics."""
    rows = []

    for strategy in df["strategy"].unique():
        sdf = df[df["strategy"] == strategy]
        csdf = case_summary[case_summary["strategy"] == strategy]

        length = compute_length_stats(sdf)
        spearman = compute_global_spearman(sdf)
        rmcorr = compute_rmcorr(sdf)

        rows.append({
            "strategy": strategy,
            "n_segments_total": len(sdf),
            "seg_length": f"{length['seg_len_mean']:.1f} ± {length['seg_len_std']:.1f}",
            "n_unique_lengths": length["n_unique_lengths"],
            "spearman_rho": (f"{spearman['spearman_rho']:.3f} (p={spearman['spearman_p']:.4f})"
                            if not np.isnan(spearman["spearman_rho"]) else "n/a"),
            "rmcorr_r": (f"{rmcorr['rmcorr_r']:.3f} (p={rmcorr['rmcorr_p']:.4f})"
                         if not np.isnan(rmcorr["rmcorr_r"]) else
                         rmcorr.get("rmcorr_note", "n/a")),
            "max_abs_sv": f"{csdf['max_abs_sv'].mean():.4f} ± {csdf['max_abs_sv'].std():.4f}",
            "sum_abs_sv": f"{csdf['sum_abs_sv'].mean():.4f} ± {csdf['sum_abs_sv'].std():.4f}",
            "mean_abs_sv": f"{csdf['mean_abs_sv'].mean():.4f} ± {csdf['mean_abs_sv'].std():.4f}",
        })

    return pd.DataFrame(rows)


# ── 5. Main ─────────────────────────────────────────────────────────

def main():
    os.makedirs(TABLE_DIR, exist_ok=True)

    print("Loading strategy results …")
    df = load_all_strategies()
    print(f"Total segments: {len(df)}\n")

    case_summary = compute_case_level_summary(df)

    # ── Summary table ───────────────────────────────────────────────
    table = build_summary_table(df, case_summary)
    print("=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(table.to_string(index=False))

    table_path = osp.join(TABLE_DIR, "strategy_comparison_table.csv")
    table.to_csv(table_path, index=False)
    print(f"\nSaved to {table_path}")

    # ── Friedman tests ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("FRIEDMAN TESTS")
    print("=" * 80)

    for metric in ["max_abs_sv", "sum_abs_sv", 'mean_abs_sv']:
        result = run_friedman_test(case_summary, metric)
        sig = "***" if result["friedman_p"] < 0.001 else (
              "**" if result["friedman_p"] < 0.01 else (
              "*" if result["friedman_p"] < 0.05 else "ns"))
        print(f"\n  Metric: {metric}")
        print(f"  χ² = {result['friedman_stat']:.4f}, "
              f"p = {result['friedman_p']:.4f} {sig}, "
              f"n = {result['n_cases']} cases")

        # ── Pairwise Wilcoxon if Friedman is significant ────────────
        if result["friedman_p"] < 0.05:
            print(f"\n  Pairwise Wilcoxon (Bonferroni-corrected):")
            pw = run_pairwise_wilcoxon(case_summary, metric)
            for _, row in pw.iterrows():
                sig_pw = "*" if row["p_bonferroni"] < 0.05 else "ns"
                print(f"    {row['pair']:30s}  W={row['W']:8.1f}  "
                      f"p_raw={row['p_raw']:.3f}  "
                      f"p_bonf={row['p_bonferroni']:.3f} {sig_pw}")

            pw_path = osp.join(TABLE_DIR, f"pairwise_wilcoxon_{metric}.csv")
            pw.to_csv(pw_path, index=False)
        else:
            print("  → Not significant, skipping pairwise comparisons.")

    # ── LaTeX table (optional) ──────────────────────────────────────
    try:
        latex = table.to_latex(index=False, escape=False, column_format="lcccccc")
        latex_path = osp.join(TABLE_DIR, "strategy_comparison_table.tex")
        with open(latex_path, "w") as f:
            f.write(latex)
        print(f"\nLaTeX table saved to {latex_path}")
    except Exception:
        pass

    print(f"\n✓ All outputs in {TABLE_DIR}")


if __name__ == "__main__":
    main()