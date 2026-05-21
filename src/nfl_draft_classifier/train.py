from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "Drafted"
ID_COL = "Id"


@dataclass(frozen=True)
class TrainResult:
    model_name: str
    mean_auc: float
    fold_aucs: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train NFL Draft classifier and create a submission file."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/input"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--model",
        choices=[
            "auto",
            "extra_trees",
            "random_forest",
            "gradient_boosting",
            "hist_gradient_boosting",
            "logistic",
        ],
        default="auto",
        help="Model to train. Use auto to compare all candidates by CV AUC.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--save-model", action="store_true")
    return parser.parse_args()


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def add_features(df: pd.DataFrame) -> pd.DataFrame:
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

    return df


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = [
        column
        for column in X.columns
        if pd.api.types.is_object_dtype(X[column].dtype)
        or pd.api.types.is_string_dtype(X[column].dtype)
        or isinstance(X[column].dtype, pd.CategoricalDtype)
    ]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("one_hot", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, numeric_cols),
            ("categorical", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def candidate_models(seed: int) -> dict[str, object]:
    return {
        "extra_trees": ExtraTreesClassifier(
            n_estimators=700,
            min_samples_leaf=3,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=600,
            max_depth=9,
            min_samples_leaf=4,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.035,
            max_depth=2,
            min_samples_leaf=12,
            subsample=0.85,
            random_state=seed,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=400,
            learning_rate=0.035,
            l2_regularization=0.05,
            max_leaf_nodes=15,
            random_state=seed,
        ),
        "logistic": LogisticRegression(
            C=0.4,
            class_weight="balanced",
            max_iter=3000,
            solver="lbfgs",
            random_state=seed,
        ),
    }


def build_pipeline(model: object, X: pd.DataFrame) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X)),
            ("model", model),
        ]
    )


def predict_probability(estimator: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    scores = estimator.decision_function(X)
    return 1 / (1 + np.exp(-scores))


def evaluate_model(
    model_name: str,
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    folds: int,
    seed: int,
) -> TrainResult:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_aucs: list[float] = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
        estimator = build_pipeline(clone(model), X)
        estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
        valid_pred = predict_probability(estimator, X.iloc[valid_idx])
        auc = roc_auc_score(y.iloc[valid_idx], valid_pred)
        fold_aucs.append(float(auc))
        print(f"{model_name} fold {fold}: AUC={auc:.5f}")

    mean_auc = float(np.mean(fold_aucs))
    print(f"{model_name} mean AUC={mean_auc:.5f}\n")
    return TrainResult(model_name=model_name, mean_auc=mean_auc, fold_aucs=fold_aucs)


def fold_ensemble_predictions(
    model: object,
    X: pd.DataFrame,
    y: pd.Series,
    test_X: pd.DataFrame,
    folds: int,
    seed: int,
) -> np.ndarray:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    predictions: list[np.ndarray] = []

    for train_idx, _ in splitter.split(X, y):
        estimator = build_pipeline(clone(model), X)
        estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
        predictions.append(predict_probability(estimator, test_X))

    return np.mean(predictions, axis=0)


def save_cv_report(results: Iterable[TrainResult], reports_dir: Path) -> None:
    rows = []
    for result in results:
        for fold, auc in enumerate(result.fold_aucs, start=1):
            rows.append(
                {
                    "model": result.model_name,
                    "fold": fold,
                    "auc": auc,
                    "mean_auc": result.mean_auc,
                }
            )
    reports_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(reports_dir / "cv_scores.csv", index=False)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    train_path = args.input_dir / "train.csv"
    test_path = args.input_dir / "test.csv"
    sample_path = args.input_dir / "sample_submission.csv"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_submission = pd.read_csv(sample_path)

    train = add_features(train)
    test = add_features(test)

    X = train.drop(columns=[TARGET, ID_COL])
    y = train[TARGET].astype(int)
    test_X = test.drop(columns=[ID_COL])

    models = candidate_models(args.seed)
    selected = models if args.model == "auto" else {args.model: models[args.model]}

    results = [
        evaluate_model(name, model, X, y, args.folds, args.seed)
        for name, model in selected.items()
    ]
    save_cv_report(results, args.reports_dir)

    best = max(results, key=lambda result: result.mean_auc)
    best_model = models[best.model_name]
    print(f"Best model: {best.model_name} ({best.mean_auc:.5f} mean AUC)")

    test_pred = fold_ensemble_predictions(
        best_model, X, y, test_X, args.folds, args.seed
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission = sample_submission.copy()
    submission[TARGET] = np.clip(test_pred, 0, 1)
    submission.to_csv(args.output_dir / "submission.csv", index=False)

    if args.save_model:
        args.models_dir.mkdir(parents=True, exist_ok=True)
        final_estimator = build_pipeline(clone(best_model), X)
        final_estimator.fit(X, y)
        joblib.dump(final_estimator, args.models_dir / f"{best.model_name}.joblib")

    print(f"Saved submission to {args.output_dir / 'submission.csv'}")
    print(f"Saved CV report to {args.reports_dir / 'cv_scores.csv'}")


if __name__ == "__main__":
    main()
