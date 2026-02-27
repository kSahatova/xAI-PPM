"""
Segment length → SV sign analysis.

Research question: Do longer segments systematically push predictions in one
direction (positive SV → toward predicted class) or the other (negative SV
→ away from predicted class)?

Three complementary analyses:
  1. Spearman & rmcorr on signed SV (not |SV|)
  2. Sign proportion test: among long vs short segments, is the ratio of
     positive-to-negative SV different?
  3. Logistic regression: does segment length predict whether SV is positive?

Usage:
    python sv_sign_analysis.py
"""

import os
import os.path as osp
import pickle

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
import matplotlib.pyplot as plt
import seaborn as sns

# ── Configuration ───────────────────────────────────────────────────
OUTPUT_ROOT = r"D:\PycharmProjects\xAI-PPM\outputs"
RESULTS_PATH = osp.join(OUTPUT_ROOT, "shap_values", "bpi17", "transition", "segment_sv_results.pkl")
SIGN_DIR = osp.join(OUTPUT_ROOT, "sv_sign_analysis")

SAMPLE_NAMES = ["tp", "tn", "fp", "fn"]


# ── 1. Load and flatten ─────────────────────────────────────────────

def load_segment_df() -> pd.DataFrame:
    with open(RESULTS_PATH, "rb") as f:
        payload = pickle.load(f)

    sv_results = payload["sv_results"]
    rows = []

    for name in SAMPLE_NAMES:
        for case_idx, res in enumerate(sv_results[name]):
            sv_arr = np.asarray(res["segment_sv"]).ravel()
            seg_ids = res["segment_ids"]

            for seg_idx, (sv_val, seg) in enumerate(zip(sv_arr, seg_ids)):
                rows.append({
                    "sample": name,
                    "case_idx": case_idx,
                    "case_key": f"{name}_{case_idx}",
                    "seg_idx": seg_idx,
                    "seg_length": len(seg),
                    "sv": float(sv_val),
                    "abs_sv": abs(float(sv_val)),
                    "sv_positive": int(float(sv_val) > 0),
                })

    return pd.DataFrame(rows)


# ── 2. Correlation on signed SV ─────────────────────────────────────

def signed_sv_correlation(df: pd.DataFrame):
    """Spearman and rmcorr between seg_length and signed SV."""
    print("=" * 70)
    print("ANALYSIS 1: Correlation with signed SV")
    print("=" * 70)

    # Global Spearman
    rho, p = sp_stats.spearmanr(df["seg_length"], df["sv"])
    print(f"\nGlobal Spearman (length vs signed SV):")
    print(f"  ρ = {rho:.4f}, p = {p:.4f}")

    # rmcorr
    try:
        import pingouin as pg
        result = pg.rm_corr(data=df, x="seg_length", y="sv", subject="case_key")
        r = result["r"].values[0]
        p_rm = result["pval"].values[0]
        print(f"\nrmcorr (length vs signed SV, controlling for case):")
        print(f"  r = {r:.4f}, p = {p_rm:.4f}")
    except ImportError:
        # Manual fallback
        df_c = df.copy()
        gm_x = df_c.groupby("case_key")["seg_length"].transform("mean")
        gm_y = df_c.groupby("case_key")["sv"].transform("mean")
        r, p_rm = sp_stats.pearsonr(df_c["seg_length"] - gm_x, df_c["sv"] - gm_y)
        print(f"\nrmcorr manual (length vs signed SV, controlling for case):")
        print(f"  r = {r:.4f}, p = {p_rm:.4f}")

    # Per bucket
    print(f"\nPer prediction bucket (Spearman):")
    for name in SAMPLE_NAMES:
        sub = df[df["sample"] == name]
        if sub["seg_length"].nunique() < 3:
            print(f"  {name.upper()}: n/a (insufficient length variance)")
            continue
        r_s, p_s = sp_stats.spearmanr(sub["seg_length"], sub["sv"])
        print(f"  {name.upper()}: ρ = {r_s:.4f}, p = {p_s:.4f}")


# ── 3. Sign proportion by length bin ────────────────────────────────

def sign_proportion_analysis(df: pd.DataFrame):
    """Compare the proportion of positive SVs in short vs long segments."""
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Proportion of positive SV by segment length")
    print("=" * 70)

    # Bin segments into short / long using median split
    median_len = df["seg_length"].median()
    df = df.copy()
    df["length_group"] = np.where(df["seg_length"] <= median_len, "short", "long")

    # Contingency table
    ct = pd.crosstab(df["length_group"], df["sv_positive"],
                      margins=True, margins_name="total")
    ct.columns = ["negative", "positive", "total"]
    print(f"\nContingency table (median split at length = {median_len}):")
    print(ct.to_string())

    # Proportions
    for group in ["short", "long"]:
        sub = df[df["length_group"] == group]
        prop = sub["sv_positive"].mean()
        print(f"  {group}: {prop:.1%} positive SV (n={len(sub)})")

    # Chi-squared test
    short = df[df["length_group"] == "short"]["sv_positive"]
    long = df[df["length_group"] == "long"]["sv_positive"]

    table = np.array([
        [short.sum(), len(short) - short.sum()],
        [long.sum(), len(long) - long.sum()],
    ])
    chi2, p_chi, dof, expected = sp_stats.chi2_contingency(table)
    print(f"\nChi-squared test (H₀: sign proportions are equal):")
    print(f"  χ² = {chi2:.4f}, p = {p_chi:.4f}, dof = {dof}")

    # Also try with tercile split for finer granularity
    try:
        df["length_bin"] = pd.qcut(df["seg_length"], q=3,
                                    labels=["short", "medium", "long"],
                                    duplicates="drop")
        print(f"\nProportion of positive SV by tercile:")
        for bin_name in ["short", "medium", "long"]:
            sub = df[df["length_bin"] == bin_name]
            if len(sub) > 0:
                print(f"  {bin_name}: {sub['sv_positive'].mean():.1%} positive "
                      f"(n={len(sub)}, mean length={sub['seg_length'].mean():.1f})")
    except ValueError:
        print("\n  (Tercile split not possible — too few unique lengths)")

    return df


# ── 4. Logistic regression ──────────────────────────────────────────

def logistic_regression_analysis(df: pd.DataFrame):
    """Does segment length predict whether SV is positive?"""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Logistic regression (length → P(SV > 0))")
    print("=" * 70)

    try:
        import statsmodels.api as sm

        X = sm.add_constant(df["seg_length"])
        y = df["sv_positive"]

        model = sm.Logit(y, X).fit(disp=0)
        print(f"\n{model.summary2().tables[1].to_string()}")

        coef = model.params["seg_length"]
        p_val = model.pvalues["seg_length"]
        odds_ratio = np.exp(coef)

        print(f"\nSegment length coefficient: {coef:.4f} (p = {p_val:.4f})")
        print(f"Odds ratio: {odds_ratio:.4f}")
        print(f"Interpretation: each additional event in a segment multiplies")
        print(f"  the odds of a positive SV by {odds_ratio:.3f}")

    except ImportError:
        print("\n  statsmodels not installed — skipping logistic regression.")
        print("  Install with: pip install statsmodels")


# ── 5. Visualisation ────────────────────────────────────────────────

def plot_signed_sv(df: pd.DataFrame):
    """Scatter and box plots for signed SV vs length."""
    plt.style.use("seaborn-v0_8-whitegrid")

    # Scatter: signed SV vs length
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.stripplot(data=df, x="seg_length", y="sv", hue="sample",
                  dodge=True, alpha=0.5, size=4, ax=ax)
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Segment length (# events)")
    ax.set_ylabel("Signed SHAP value")
    ax.set_title("Signed SV vs. segment length")
    fig.tight_layout()
    fig.savefig(osp.join(SIGN_DIR, "scatter_signed_sv_vs_length.png"), dpi=300)
    plt.close()

    # Box plot: SV distribution per length
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="seg_length", y="sv", ax=ax)
    ax.axhline(y=0, color="red", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Segment length (# events)")
    ax.set_ylabel("Signed SHAP value")
    ax.set_title("SV distribution by segment length")
    fig.tight_layout()
    fig.savefig(osp.join(SIGN_DIR, "boxplot_signed_sv_by_length.png"), dpi=300)
    plt.close()

    # Proportion plot
    prop = (df.groupby("seg_length")["sv_positive"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "prop_positive", "count": "n"}))
    prop = prop[prop["n"] >= 3]  # only lengths with enough data

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(prop.index, prop["prop_positive"], color="steelblue", edgecolor="white")
    ax.axhline(y=0.5, color="red", linestyle="--", linewidth=0.8, label="50% baseline")
    for idx, row in prop.iterrows():
        ax.text(idx, row["prop_positive"] + 0.02, f'n={int(row["n"])}',
                ha="center", fontsize=9)
    ax.set_xlabel("Segment length (# events)")
    ax.set_ylabel("Proportion of positive SV")
    ax.set_title("Proportion of positive SV by segment length")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(osp.join(SIGN_DIR, "proportion_positive_sv_by_length.png"), dpi=300)
    plt.close()

    print(f"\nPlots saved to {SIGN_DIR}")


# ── Main ────────────────────────────────────────────────────────────

def main():
    os.makedirs(SIGN_DIR, exist_ok=True)

    print("Loading data …\n")
    df = load_segment_df()
    print(f"Total segments: {len(df)}")
    print(f"Positive SV: {df['sv_positive'].sum()} ({df['sv_positive'].mean():.1%})")
    print(f"Negative SV: {(1 - df['sv_positive']).sum()} ({1 - df['sv_positive'].mean():.1%})\n")

    signed_sv_correlation(df)
    df = sign_proportion_analysis(df)
    logistic_regression_analysis(df)
    plot_signed_sv(df)

    print(f"\n✓ SV sign analysis complete. Outputs in {SIGN_DIR}")


if __name__ == "__main__":
    main()