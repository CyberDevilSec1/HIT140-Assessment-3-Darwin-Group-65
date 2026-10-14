

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
OUT = BASE_DIR / "outputs" / "objective2"
OUT.mkdir(parents=True, exist_ok=True)

# Pre-specified grid used for the PRIMARY model comparison.
PRIMARY_ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]


def metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def date_group_split(df, key, test_fraction=0.20):
    """Hold out complete final dates, targeting about 20% of unique matches."""
    match_dates = df[[key, "date"]].drop_duplicates()
    if match_dates[key].duplicated().any():
        raise ValueError("A match has inconsistent dates.")

    match_dates = match_dates.assign(
        date=pd.to_datetime(match_dates["date"], errors="raise").dt.normalize()
    )
    dates = sorted(match_dates["date"].unique())
    if len(dates) < 6:
        raise ValueError("Not enough distinct dates for a date-grouped split.")

    counts = match_dates.groupby("date").size()
    n_matches = len(match_dates)
    candidate_cuts = range(1, len(dates))
    cut = min(
        candidate_cuts,
        key=lambda i: (
            abs(int(counts.loc[dates[i:]].sum()) - n_matches * test_fraction),
            -i,
        ),
    )

    train_dates = set(dates[:cut])
    test_dates = set(dates[cut:])
    normalized = pd.to_datetime(df["date"], errors="raise").dt.normalize()
    train = df.loc[normalized.isin(train_dates)].copy()
    test = df.loc[normalized.isin(test_dates)].copy()

    if train.empty or test.empty:
        raise ValueError("Chronological train/test split produced an empty set.")
    if set(train[key]).intersection(set(test[key])):
        raise ValueError("A match appears in both training and testing sets.")
    if pd.to_datetime(train["date"]).max() >= pd.to_datetime(test["date"]).min():
        raise ValueError("Chronological train/test ordering failed.")

    return train, test


def expanding_date_folds(train):
    """Create three expanding chronological CV folds using complete dates."""
    dates = sorted(pd.to_datetime(train["date"], errors="raise").dt.normalize().unique())
    n_dates = len(dates)
    bounds = [int(n_dates * x) for x in (0.40, 0.55, 0.70, 0.85)]

    if len(set(bounds)) != 4 or bounds[0] < 1 or bounds[-1] >= n_dates:
        raise ValueError(f"Insufficient distinct training dates for three folds: {n_dates}")

    normalized = pd.to_datetime(train["date"], errors="raise").dt.normalize()
    folds = []
    for i in range(3):
        tr = train.loc[normalized.isin(dates[: bounds[i]])].copy()
        val = train.loc[normalized.isin(dates[bounds[i]: bounds[i + 1]])].copy()
        if tr.empty or val.empty:
            raise ValueError("A CV training or validation fold is empty.")
        if pd.to_datetime(tr["date"]).max() >= pd.to_datetime(val["date"]).min():
            raise ValueError("Chronological CV ordering failed.")
        folds.append((tr, val))
    return folds


def estimator(name, alpha=None):
    if name == "Baseline":
        return DummyRegressor(strategy="mean")
    if name == "OLS":
        return make_pipeline(StandardScaler(), LinearRegression())
    if name == "Ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    if name == "Lasso":
        return make_pipeline(StandardScaler(), Lasso(alpha=alpha, max_iter=100000))
    raise ValueError(f"Unknown model: {name}")


def _task_config(task):
    if task not in ("2_1", "2_2"):
        raise ValueError("Choose task '2_1' or '2_2'.")
    is_team_task = task == "2_2"
    return {
        "is_team_task": is_team_task,
        "filename": "objective2_2_dataset_208.csv" if is_team_task else "objective2_1_dataset_104.csv",
        "target": "goals_scored" if is_team_task else "goal_difference_team_a",
        "key": "match_id" if is_team_task else "record_id",
        "expected_rows": 208 if is_team_task else 104,
    }


def load_task_data(task):
    """Load and validate a prepared modelling dataset."""
    cfg = _task_config(task)
    data_file = OUT / cfg["filename"]
    if not data_file.is_file():
        raise FileNotFoundError(
            f"Missing prepared dataset: {data_file}\n"
            "Run 05_objective2_prepare_datasets_CORRECTED.py first."
        )

    df = pd.read_csv(data_file)
    features = [c for c in df.columns if "_prior_" in c]
    target = cfg["target"]
    key = cfg["key"]

    if len(df) != cfg["expected_rows"]:
        raise ValueError(f"Expected {cfg['expected_rows']} rows, found {len(df)}.")
    if len(features) != 8:
        raise ValueError(f"Expected exactly 8 explanatory variables, found {len(features)}: {features}")
    if df[features + [target, "date", key]].isna().any().any():
        raise ValueError("Missing model values, dates or match identifiers detected.")
    if df[key].nunique() != 104:
        raise ValueError("Expected exactly 104 distinct match identifiers.")
    if cfg["is_team_task"] and not df.groupby(key).size().eq(2).all():
        raise ValueError("Each Regression 2.2 match must contain exactly two team rows.")
    if not cfg["is_team_task"] and df[key].duplicated().any():
        raise ValueError("Regression 2.1 contains duplicate match rows.")

    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.strftime("%Y-%m-%d")
    df = df.sort_values(["date", key], kind="stable").reset_index(drop=True)
    return df, features, target, key


def _save_split_manifest(task, train, test, key):
    train_unique = train[[key, "date"]].drop_duplicates().assign(partition="train")
    test_unique = test[[key, "date"]].drop_duplicates().assign(partition="test")
    manifest = pd.concat([train_unique, test_unique], ignore_index=True).sort_values(["date", key])
    manifest.to_csv(OUT / f"objective{task}_split_manifest.csv", index=False)


def _save_fold_manifest(task, folds, key):
    rows = []
    for fold_id, (tr, val) in enumerate(folds, start=1):
        rows.append({
            "fold": fold_id,
            "train_matches": tr[key].nunique(),
            "validation_matches": val[key].nunique(),
            "train_rows": len(tr),
            "validation_rows": len(val),
            "train_start": pd.to_datetime(tr["date"]).min().date().isoformat(),
            "train_end": pd.to_datetime(tr["date"]).max().date().isoformat(),
            "validation_start": pd.to_datetime(val["date"]).min().date().isoformat(),
            "validation_end": pd.to_datetime(val["date"]).max().date().isoformat(),
        })
    pd.DataFrame(rows).to_csv(OUT / f"objective{task}_cv_fold_manifest.csv", index=False)


def run(task):
    """Run the primary Objective 2 model comparison for one task."""
    df, features, target, key = load_task_data(task)
    train, test = date_group_split(df, key)
    folds = expanding_date_folds(train)
    _save_split_manifest(task, train, test, key)
    _save_fold_manifest(task, folds, key)

    # Candidate models evaluated by training-period CV only.
    specs = [("Baseline", None), ("OLS", None)]
    specs += [
        (name, alpha)
        for name in ("Ridge", "Lasso")
        for alpha in PRIMARY_ALPHA_GRID
    ]

    cv_rows = []
    for name, alpha in specs:
        for fold_id, (tr, val) in enumerate(folds, start=1):
            if set(tr[key]).intersection(set(val[key])):
                raise ValueError("A match appears in both CV training and validation sets.")

            model = estimator(name, alpha)
            model.fit(tr[features], tr[target])
            pred = model.predict(val[features])
            score = metrics(val[target], pred)
            cv_rows.append({
                "model": name,
                "alpha": alpha,
                "fold": fold_id,
                "training_matches": tr[key].nunique(),
                "validation_matches": val[key].nunique(),
                **score,
            })

    cv = pd.DataFrame(cv_rows)
    cv.to_csv(OUT / f"objective{task}_cv_all_folds.csv", index=False)

    # Mean + variability across folds strengthens the stability analysis.
    mean_summary = (
        cv.groupby(["model", "alpha"], dropna=False)[["MAE", "RMSE", "R2"]]
        .mean()
        .reset_index()
    )
    sd_summary = (
        cv.groupby(["model", "alpha"], dropna=False)[["MAE", "RMSE", "R2"]]
        .std(ddof=1)
        .reset_index()
        .rename(columns={"MAE": "MAE_SD", "RMSE": "RMSE_SD", "R2": "R2_SD"})
    )
    summary = mean_summary.merge(sd_summary, on=["model", "alpha"], how="left")
    summary.to_csv(OUT / f"objective{task}_cv_summary.csv", index=False)

    # Select Ridge/Lasso alpha by mean CV MAE from the primary grid.
    selected = [("Baseline", None), ("OLS", None)]
    for name in ("Ridge", "Lasso"):
        options = summary.loc[summary["model"].eq(name)].sort_values(
            ["MAE", "alpha"], ascending=[True, True]
        )
        selected.append((name, float(options.iloc[0]["alpha"])))

    results = []
    prediction_table = test[[key, "date", "stage", target]].copy()
    coefficient_tables = {}

    for name, alpha in selected:
        model = estimator(name, alpha)
        model.fit(train[features], train[target])
        predictions = model.predict(test[features])
        prediction_table[f"predicted_{name.lower()}"] = predictions

        score = metrics(test[target], predictions)
        cv_score = summary.loc[summary["model"].eq(name)]
        if alpha is not None:
            cv_score = cv_score.loc[cv_score["alpha"].eq(alpha)]

        results.append({
            "model": name,
            "selected_alpha": alpha,
            "cv_MAE": float(cv_score.iloc[0]["MAE"]),
            "cv_MAE_SD": float(cv_score.iloc[0]["MAE_SD"]),
            "cv_RMSE": float(cv_score.iloc[0]["RMSE"]),
            "cv_RMSE_SD": float(cv_score.iloc[0]["RMSE_SD"]),
            "train_rows": len(train),
            "test_rows": len(test),
            **score,
        })

        # Coefficients in both standardized and original units.
        if name != "Baseline":
            scaler = model.named_steps["standardscaler"]
            linear_model = model.steps[-1][1]
            standardized = np.asarray(linear_model.coef_, dtype=float)
            original = standardized / scaler.scale_
            coefficients = pd.DataFrame({
                "feature": features,
                "coefficient_standardized": standardized,
                "coefficient_original_units": original,
            })
            coefficients["abs_standardized"] = coefficients["coefficient_standardized"].abs()
            coefficients["abs_original_units"] = coefficients["coefficient_original_units"].abs()
            coefficients = coefficients.sort_values("abs_standardized", ascending=False)
            coefficients.to_csv(
                OUT / f"objective{task}_{name.lower()}_coefficients.csv", index=False
            )
            coefficient_tables[name] = coefficients

    results = pd.DataFrame(results)
    results.to_csv(OUT / f"objective{task}_test_model_comparison.csv", index=False)
    prediction_table.to_csv(OUT / f"objective{task}_test_predictions_all_models.csv", index=False)

    # Preferred non-baseline algorithm is selected by mean CV MAE ONLY.
    candidates = results.loc[results["model"].ne("Baseline")].sort_values(
        ["cv_MAE", "model"], ascending=[True, True]
    )
    chosen = str(candidates.iloc[0]["model"])
    chosen_alpha = candidates.iloc[0]["selected_alpha"]

    # Save chosen coefficients as a dedicated file for report reproducibility.
    if chosen in coefficient_tables:
        coefficient_tables[chosen].to_csv(
            OUT / f"objective{task}_selected_model_coefficients.csv", index=False
        )

    pred = prediction_table[f"predicted_{chosen.lower()}"].to_numpy()
    actual = test[target].to_numpy()

    # ------------------------------------------------------------------
    # Visual 1: CV vs held-out MAE (shows selection vs final evaluation)
    # ------------------------------------------------------------------
    x = np.arange(len(results))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - width/2, results["cv_MAE"], width, label="Mean CV MAE")
    ax.bar(x + width/2, results["MAE"], width, label="Held-out MAE")
    ax.set_xticks(x, results["model"])
    ax.set_ylabel("MAE (lower is better)")
    ax.set_title(f"Regression {task.replace('_', '.')}: CV selection vs held-out error")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / f"objective{task}_cv_vs_test_mae.png", dpi=220)
    plt.close(fig)

    # Retain the simpler held-out MAE chart for continuity.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(results["model"], results["MAE"])
    ax.set_ylabel("Test MAE (lower is better)")
    ax.set_title(f"Regression {task.replace('_', '.')}: held-out model comparison")
    fig.tight_layout()
    fig.savefig(OUT / f"objective{task}_model_comparison.png", dpi=220)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Visual 2: actual vs predicted for CV-selected model
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(actual, pred, alpha=0.75)
    lo = min(actual.min(), pred.min())
    hi = max(actual.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "--", color="gray")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Regression {task.replace('_', '.')}: {chosen} selected by CV")
    fig.tight_layout()
    fig.savefig(OUT / f"objective{task}_cv_selected_actual_vs_predicted.png", dpi=220)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Visual 3: residual diagnostics
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(pred, actual - pred, alpha=0.75)
    ax.axhline(0, linestyle="--", color="gray")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title(f"Regression {task.replace('_', '.')}: residual diagnostics")
    fig.tight_layout()
    fig.savefig(OUT / f"objective{task}_cv_selected_residuals.png", dpi=220)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Visual 4: fold-by-fold stability for selected candidates
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name, alpha in selected:
        sub = cv.loc[cv["model"].eq(name)].copy()
        if alpha is not None:
            sub = sub.loc[sub["alpha"].eq(alpha)]
        ax.plot(sub["fold"], sub["MAE"], marker="o", label=name)
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("Expanding CV fold")
    ax.set_ylabel("Validation MAE")
    ax.set_title(f"Regression {task.replace('_', '.')}: fold-level stability")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / f"objective{task}_cv_fold_stability.png", dpi=220)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Visual 5: standardized coefficients of selected model
    # ------------------------------------------------------------------
    if chosen in coefficient_tables:
        coef_plot = coefficient_tables[chosen].sort_values("coefficient_standardized")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(coef_plot["feature"], coef_plot["coefficient_standardized"])
        ax.axvline(0, linewidth=1)
        ax.set_xlabel("Standardized coefficient")
        ax.set_title(
            f"Regression {task.replace('_', '.')}: {chosen} feature effects "
            f"(CV-selected)"
        )
        ax.tick_params(axis="y", labelsize=8)
        fig.tight_layout()
        fig.savefig(OUT / f"objective{task}_selected_standardized_coefficients.png", dpi=220)
        plt.close(fig)

    # ------------------------------------------------------------------
    # Terminal summary
    # ------------------------------------------------------------------
    print(
        f"\nRegression {task.replace('_', '.')}: {len(df)} rows; "
        f"{len(features)} predictors; {len(train)} train / {len(test)} test rows"
    )
    print(
        f"Date-grouped split: train through {train['date'].max()}, "
        f"test from {test['date'].min()}"
    )
    print("Expanding CV grouped by complete dates; test untouched during selection:")
    display_cols = [
        "model", "selected_alpha", "cv_MAE", "cv_MAE_SD", "cv_RMSE",
        "train_rows", "test_rows", "MAE", "RMSE", "R2"
    ]
    print(results[display_cols].round(4).to_string(index=False))
    print(
        f"CV-selected non-baseline model: {chosen}"
        + (f" (alpha={chosen_alpha:g})" if pd.notna(chosen_alpha) else "")
        + "; held-out test results are descriptive, not a tuning criterion."
    )

    return results


if __name__ == "__main__":
    print("=" * 72)
    print("HIT140 OBJECTIVE 2 - PRIMARY MODEL COMPARISON")
    print("=" * 72)
    run("2_1")
    run("2_2")
