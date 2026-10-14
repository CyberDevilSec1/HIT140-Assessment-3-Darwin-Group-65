"""HIT140 Objective 2 - HD diagnostics and training-only sensitivity analysis.

IMPORTANT SCIENTIFIC NOTE
-------------------------
The primary models and primary alpha grid are defined in
09_objective2_model_comparison.py. This script is an additional robustness check.
It NEVER uses the held-out test set to choose or tune a model.

Because the held-out test results have already been inspected, the extended alpha
search here must be described as an exploratory TRAINING-ONLY sensitivity analysis.
It strengthens the discussion of model stability but does not create a new blind test.
"""

from pathlib import Path
from importlib import import_module
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / "outputs" / "objective2"
OUT.mkdir(parents=True, exist_ok=True)

primary = import_module("09_objective2_model_comparison")

# Extended grid used ONLY for training-CV sensitivity.
SENSITIVITY_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0, 300.0, 1000.0]


def _primary_selected_model(task):
    """Identify the primary CV-selected non-baseline model from 09 outputs."""
    result_file = OUT / f"objective{task}_test_model_comparison.csv"
    if not result_file.is_file():
        raise FileNotFoundError(
            f"Missing {result_file.name}. Run 09_objective2_model_comparison.py first."
        )
    results = pd.read_csv(result_file)
    candidates = results.loc[results["model"].ne("Baseline")].sort_values(
        ["cv_MAE", "model"], ascending=[True, True]
    )
    row = candidates.iloc[0]
    alpha = None if pd.isna(row["selected_alpha"]) else float(row["selected_alpha"])
    return str(row["model"]), alpha


def regularisation_sensitivity(task):
    """Evaluate Ridge/Lasso alpha sensitivity using training-period CV only."""
    df, features, target, key = primary.load_task_data(task)
    train, _test = primary.date_group_split(df, key)
    folds = primary.expanding_date_folds(train)

    rows = []
    for model_name in ("Ridge", "Lasso"):
        for alpha in SENSITIVITY_ALPHAS:
            fold_scores = []
            for fold_id, (tr, val) in enumerate(folds, start=1):
                model = primary.estimator(model_name, alpha)
                model.fit(tr[features], tr[target])
                pred = model.predict(val[features])
                score = primary.metrics(val[target], pred)
                rows.append({
                    "model": model_name,
                    "alpha": alpha,
                    "fold": fold_id,
                    **score,
                })
                fold_scores.append(score)

    all_folds = pd.DataFrame(rows)
    all_folds.to_csv(
        OUT / f"objective{task}_training_only_regularisation_sensitivity_folds.csv",
        index=False,
    )

    summary = (
        all_folds.groupby(["model", "alpha"], as_index=False)
        .agg(
            cv_MAE=("MAE", "mean"),
            cv_MAE_SD=("MAE", "std"),
            cv_RMSE=("RMSE", "mean"),
            cv_RMSE_SD=("RMSE", "std"),
            cv_R2=("R2", "mean"),
            cv_R2_SD=("R2", "std"),
        )
    )
    summary.to_csv(
        OUT / f"objective{task}_training_only_regularisation_sensitivity.csv",
        index=False,
    )

    # One clear sensitivity chart: alpha vs mean CV MAE.
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for model_name in ("Ridge", "Lasso"):
        sub = summary.loc[summary["model"].eq(model_name)].sort_values("alpha")
        ax.plot(sub["alpha"], sub["cv_MAE"], marker="o", label=model_name)
    ax.set_xscale("log")
    ax.set_xlabel("Regularisation alpha (log scale)")
    ax.set_ylabel("Mean training-period CV MAE")
    ax.set_title(
        f"Regression {task.replace('_', '.')}: training-only regularisation sensitivity"
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(
        OUT / f"objective{task}_training_only_regularisation_sensitivity.png",
        dpi=220,
    )
    plt.close(fig)

    return summary


def coefficient_stability(task):
    """Assess standardized coefficient stability across expanding CV training windows."""
    df, features, target, key = primary.load_task_data(task)
    train, _test = primary.date_group_split(df, key)
    folds = primary.expanding_date_folds(train)
    chosen, alpha = _primary_selected_model(task)

    if chosen == "Baseline":
        return None

    rows = []
    for fold_id, (tr, _val) in enumerate(folds, start=1):
        model = primary.estimator(chosen, alpha)
        model.fit(tr[features], tr[target])
        linear_model = model.steps[-1][1]
        coefs = np.asarray(linear_model.coef_, dtype=float)
        for feature, coef in zip(features, coefs):
            rows.append({
                "fold": fold_id,
                "model": chosen,
                "alpha": alpha,
                "feature": feature,
                "standardized_coefficient": float(coef),
            })

    fold_coefs = pd.DataFrame(rows)
    fold_coefs.to_csv(
        OUT / f"objective{task}_selected_model_coefficient_stability_folds.csv",
        index=False,
    )

    stability = (
        fold_coefs.groupby("feature", as_index=False)
        .agg(
            mean_standardized_coefficient=("standardized_coefficient", "mean"),
            sd_standardized_coefficient=("standardized_coefficient", "std"),
            min_standardized_coefficient=("standardized_coefficient", "min"),
            max_standardized_coefficient=("standardized_coefficient", "max"),
        )
    )
    stability["abs_mean"] = stability["mean_standardized_coefficient"].abs()
    stability = stability.sort_values("abs_mean", ascending=False)
    stability.to_csv(
        OUT / f"objective{task}_selected_model_coefficient_stability.csv",
        index=False,
    )

    # Coefficient means with fold-to-fold SD as uncertainty bars.
    plot_df = stability.sort_values("mean_standardized_coefficient")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(
        plot_df["mean_standardized_coefficient"],
        plot_df["feature"],
        xerr=plot_df["sd_standardized_coefficient"].fillna(0),
        fmt="o",
        capsize=3,
    )
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Mean standardized coefficient across CV training windows")
    ax.set_title(
        f"Regression {task.replace('_', '.')}: {chosen} coefficient stability"
    )
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(
        OUT / f"objective{task}_selected_model_coefficient_stability.png",
        dpi=220,
    )
    plt.close(fig)

    return stability


def print_sensitivity_summary(task, summary):
    print("\n" + "-" * 72)
    print(f"Regression {task.replace('_', '.')} - TRAINING-ONLY robustness checks")
    print("-" * 72)
    for model_name in ("Ridge", "Lasso"):
        sub = summary.loc[summary["model"].eq(model_name)].sort_values(
            ["cv_MAE", "alpha"], ascending=[True, True]
        )
        best = sub.iloc[0]
        print(
            f"Sensitivity minimum for {model_name}: alpha={best['alpha']:g}, "
            f"CV MAE={best['cv_MAE']:.4f} (SD={best['cv_MAE_SD']:.4f})"
        )
    chosen, alpha = _primary_selected_model(task)
    print(
        f"Primary reported model from script 09 remains: {chosen}"
        + (f" (alpha={alpha:g})" if alpha is not None else "")
    )
    print(
        "Interpretation rule: this sensitivity analysis uses training CV only and "
        "must not be presented as a new untouched-test selection."
    )
    


def main():
    print("=" * 72)
    print("HIT140 OBJECTIVE 2 - HD ROBUSTNESS DIAGNOSTICS")
    print("=" * 72)
    print("No held-out test outcomes are used for sensitivity tuning in this script.")

    for task in ("2_1", "2_2"):
        summary = regularisation_sensitivity(task)
        coefficient_stability(task)
        print_sensitivity_summary(task, summary)


if __name__ == "__main__":
    main()
