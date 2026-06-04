from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


TARGET = "Drafted"
ID_COL = "Id"
CAT_COLS = ["School", "Player_Type", "Position_Type", "Position"]
MISSING_FLAG_COLS = [
    "Age",
    "Sprint_40yd",
    "Vertical_Jump",
    "Bench_Press_Reps",
    "Broad_Jump",
    "Agility_3cone",
    "Shuttle",
]
POSITION_RELATIVE_METRICS = [
    "Age",
    "Height",
    "Weight",
    "Sprint_40yd",
    "Vertical_Jump",
    "Bench_Press_Reps",
    "Broad_Jump",
    "Agility_3cone",
    "Shuttle",
    "BMI",
    "Speed_Score",
    "Explosive_Index",
    "Agility_Index",
]
SPLIT_SEEDS = [99, 42, 7]
MODEL_SEED = 2025


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the current best CatBoost NFL Draft submission."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--submission-name",
        default="submission_public_0_84999_catboost_position_features_avg_top3.csv",
    )
    return parser.parse_args()


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["BMI"] = safe_divide(df["Weight"], df["Height"] ** 2)
    df["Weight_per_Height"] = safe_divide(df["Weight"], df["Height"])
    df["Broad_per_Height"] = safe_divide(df["Broad_Jump"], df["Height"])
    df["Vertical_per_Height"] = safe_divide(df["Vertical_Jump"], df["Height"])
    df["Bench_per_Weight"] = safe_divide(df["Bench_Press_Reps"], df["Weight"])
    df["Speed_Score"] = safe_divide(df["Weight"], df["Sprint_40yd"] ** 4)
    df["Power_Speed"] = safe_divide(df["Weight"], df["Sprint_40yd"])
    df["Jump_Power"] = df["Weight"] * df["Broad_Jump"]
    df["Explosive_Index"] = df["Vertical_Jump"] + df["Broad_Jump"]
    df["Agility_Index"] = df["Agility_3cone"] + df["Shuttle"]
    df["Agility_per_Weight"] = safe_divide(df["Agility_Index"], df["Weight"])
    df["Speed_x_Broad"] = safe_divide(df["Broad_Jump"], df["Sprint_40yd"])
    df["Speed_x_Vertical"] = safe_divide(df["Vertical_Jump"], df["Sprint_40yd"])
    df["Mass_Explosive"] = df["Weight"] * df["Explosive_Index"]
    return df


def add_missing_flags(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    for col in MISSING_FLAG_COLS:
        train[f"{col}_missing"] = train[col].isna().astype(int)
        test[f"{col}_missing"] = test[col].isna().astype(int)
    return train, test


def add_position_type_relative_features(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    group_col = "Position_Type"
    global_median = train[POSITION_RELATIVE_METRICS].median(numeric_only=True)
    grouped = train.groupby(group_col, dropna=False)[POSITION_RELATIVE_METRICS].agg(
        ["median", "std"]
    )

    for col in POSITION_RELATIVE_METRICS:
        medians = grouped[(col, "median")]
        stds = grouped[(col, "std")].replace(0, np.nan)
        fallback_std = train[col].std()

        train_group_median = train[group_col].map(medians).fillna(global_median[col])
        test_group_median = test[group_col].map(medians).fillna(global_median[col])
        train_group_std = train[group_col].map(stds).fillna(fallback_std)
        test_group_std = test[group_col].map(stds).fillna(fallback_std)

        train[f"{col}_vs_{group_col}"] = train[col] - train_group_median
        test[f"{col}_vs_{group_col}"] = test[col] - test_group_median
        train[f"{col}_z_{group_col}"] = (train[col] - train_group_median) / train_group_std
        test[f"{col}_z_{group_col}"] = (test[col] - test_group_median) / test_group_std

    return train, test


def prepare_data(input_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(input_dir / "train.csv")
    test = pd.read_csv(input_dir / "test.csv")
    sample_submission = pd.read_csv(input_dir / "sample_submission.csv")

    train = add_base_features(train)
    test = add_base_features(test)
    train, test = add_missing_flags(train, test)
    train, test = add_position_type_relative_features(train, test)

    X = train.drop(columns=[ID_COL, TARGET]).copy()
    y = train[TARGET].astype(int)
    X_test = test.drop(columns=[ID_COL]).copy()

    for col in CAT_COLS:
        X[col] = X[col].astype(str).fillna("missing")
        X_test[col] = X_test[col].astype(str).fillna("missing")

    return X, y, X_test, sample_submission


def make_model() -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=2500,
        od_type="Iter",
        od_wait=120,
        random_seed=MODEL_SEED,
        allow_writing_files=False,
        depth=4,
        learning_rate=0.02,
        l2_leaf_reg=10,
        bootstrap_type="Bayesian",
        bagging_temperature=1.0,
    )


def predict_for_split_seed(
    X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame, split_seed: int
) -> np.ndarray:
    cat_feature_indices = [X.columns.get_loc(col) for col in CAT_COLS]
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=split_seed)
    test_fold_predictions: list[np.ndarray] = []

    for train_idx, valid_idx in splitter.split(X, y):
        model = make_model()
        train_pool = Pool(
            X.iloc[train_idx], y.iloc[train_idx], cat_features=cat_feature_indices
        )
        valid_pool = Pool(
            X.iloc[valid_idx], y.iloc[valid_idx], cat_features=cat_feature_indices
        )
        test_pool = Pool(X_test, cat_features=cat_feature_indices)
        model.fit(train_pool, eval_set=valid_pool, verbose=False, use_best_model=True)
        test_fold_predictions.append(model.predict_proba(test_pool)[:, 1])

    return np.mean(test_fold_predictions, axis=0)


def main() -> None:
    args = parse_args()
    np.random.seed(MODEL_SEED)

    X, y, X_test, sample_submission = prepare_data(args.input_dir)
    split_predictions = [
        predict_for_split_seed(X, y, X_test, split_seed) for split_seed in SPLIT_SEEDS
    ]
    final_prediction = np.clip(np.mean(split_predictions, axis=0), 0, 1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission = sample_submission.copy()
    submission[TARGET] = final_prediction

    submission_path = args.output_dir / args.submission_name
    submission.to_csv(submission_path, index=False)
    submission.to_csv(args.output_dir / "submission.csv", index=False)

    print(f"Saved {submission_path}")
    print(f"Saved {args.output_dir / 'submission.csv'}")


if __name__ == "__main__":
    main()
