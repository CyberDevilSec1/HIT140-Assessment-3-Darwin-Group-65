# ============================================================
# HIT140 - FIFA World Cup 2026
# Objective 2 - Exploratory Data Analysis (HD-ready)
# ============================================================
# Produces reproducible descriptive summaries and visualisations for
# both regression tasks. All analysis uses the prepared pre-match
# datasets created by 05_objective2_prepare_datasets_CORRECTED.py.
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / "outputs" / "objective2"
OUT.mkdir(parents=True, exist_ok=True)


def _top_predictor_pairs(df, features):
    """Return unique predictor-predictor correlations ordered by |r|."""
    corr = df[features].corr(numeric_only=True)
    rows = []
    for i, a in enumerate(features):
        for b in features[i + 1:]:
            rows.append({
                "feature_1": a,
                "feature_2": b,
                "correlation": float(corr.loc[a, b]),
                "abs_correlation": float(abs(corr.loc[a, b])),
            })
    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False)


def run_eda(task):
    if task not in ("2_1", "2_2"):
        raise ValueError("Choose task '2_1' or '2_2'.")

    is_team_task = task == "2_2"
    filename = "objective2_2_dataset_208.csv" if is_team_task else "objective2_1_dataset_104.csv"
    target = "goals_scored" if is_team_task else "goal_difference_team_a"
    expected_rows = 208 if is_team_task else 104
    file_path = OUT / filename

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Missing prepared dataset: {file_path}\n"
            "Run 05_objective2_prepare_datasets_CORRECTED.py first."
        )

    df = pd.read_csv(file_path)
    features = [c for c in df.columns if "_prior_" in c]

    if len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(df)}.")
    if len(features) != 8:
        raise ValueError(f"Expected exactly 8 predictors, found {len(features)}.")
    if df[features + [target]].isna().any().any():
        raise ValueError("Missing model values detected.")

    # ------------------------------------------------------------------
    # 1. Missingness and descriptive statistics
    # ------------------------------------------------------------------
    missing = df[features + [target]].isna().sum().rename("missing_count")
    missing.to_csv(OUT / f"objective{task}_missing.csv", header=True)

    descriptive = df[features + [target]].describe().T
    descriptive.to_csv(OUT / f"objective{task}_descriptive.csv")

    if "stage" in df.columns:
        stage_summary = (
            df.groupby("stage")[target]
            .agg(["count", "mean", "median", "std", "min", "max"])
            .sort_values("count", ascending=False)
        )
        stage_summary.to_csv(OUT / f"objective{task}_stage_target_summary.csv")

    # ------------------------------------------------------------------
    # 2. Correlation analysis
    # ------------------------------------------------------------------
    corr = df[features + [target]].corr(numeric_only=True)
    corr.to_csv(OUT / f"objective{task}_correlations.csv")

    target_corr = (
        corr[target]
        .drop(target)
        .rename("correlation_with_target")
        .to_frame()
    )
    target_corr["abs_correlation"] = target_corr["correlation_with_target"].abs()
    target_corr = target_corr.sort_values("abs_correlation", ascending=False)
    target_corr.to_csv(OUT / f"objective{task}_target_correlation_ranking.csv")

    pair_corr = _top_predictor_pairs(df, features)
    pair_corr.to_csv(OUT / f"objective{task}_predictor_redundancy.csv", index=False)

    # ------------------------------------------------------------------
    # 3. Target distribution
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.arange(df[target].min() - 0.5, df[target].max() + 1.5, 1)
    ax.hist(df[target], bins=bins, edgecolor="black")
    ax.axvline(df[target].mean(), linestyle="--", linewidth=1.5, label=f"Mean = {df[target].mean():.2f}")
    ax.set_xlabel(target.replace("_", " ").title())
    ax.set_ylabel("Frequency")
    ax.set_title(f"Regression {task.replace('_', '.')}: target distribution")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / f"objective{task}_target_distribution.png", dpi=220)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 4. Full correlation heatmap
    # ------------------------------------------------------------------
    matrix = corr.to_numpy()
    labels = corr.columns.tolist()
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(matrix, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(f"Regression {task.replace('_', '.')}: correlation structure")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    fig.tight_layout()
    fig.savefig(OUT / f"objective{task}_correlation_heatmap.png", dpi=220)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5. Ranked predictor-target correlations
    # ------------------------------------------------------------------
    plot_df = target_corr.sort_values("correlation_with_target")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(plot_df.index, plot_df["correlation_with_target"])
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Correlation with target (Pearson r)")
    ax.set_title(f"Regression {task.replace('_', '.')}: predictor-target associations")
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"objective{task}_target_correlation_ranking.png", dpi=220)
    plt.close(fig)

    strongest = target_corr.iloc[0]
    strongest_feature = target_corr.index[0]
    strongest_pair = pair_corr.iloc[0]

    print(
        f"EDA {task}: {len(df)} rows, {len(features)} predictors, 0 missing model values"
    )
    print(
        f"  Strongest predictor-target correlation: {strongest_feature} "
        f"(r={strongest['correlation_with_target']:.3f})"
    )
    print(
        f"  Strongest predictor-predictor correlation: "
        f"{strongest_pair['feature_1']} vs {strongest_pair['feature_2']} "
        f"(r={strongest_pair['correlation']:.3f})"
    )


if __name__ == "__main__":
    run_eda("2_1")
    run_eda("2_2")
