"""Train churn prediction models on the cleaned churn dataset.

Run this script directly from the command line:

    python src/train_model.py

What it does:
- loads ``outputs/cleaned_churn_data.csv``
- reconstructs the original raw-style features from the cleaned/encoded file
- splits the data into train and test sets with stratification
- trains a Logistic Regression baseline
- trains a Random Forest model as the stronger candidate
- evaluates both models on the test set
- runs 5-fold stratified cross-validation on the Random Forest model
- prints a comparison table and the top 10 Random Forest feature importances
- saves the best-performing trained model bundle to ``models/churn_model.pkl``
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DATA_PATH = PROJECT_ROOT / "outputs" / "cleaned_churn_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_OUTPUT_PATH = MODEL_DIR / "churn_model.pkl"

TARGET_COLUMN = "churn_status"
ID_COLUMN = "customer_id"

NUMERIC_FEATURES = [
    "subscription_length_months",
    "customer_satisfaction_score",
    "daily_watch_time_hours",
    "engagement_rate",
    "support_queries_logged",
    "age",
    "monthly_income_usd",
    "promotional_offers_used",
    "number_of_profiles_created",
]

ORDINAL_FEATURES = ["subscription_plan"]
NOMINAL_FEATURES = [
    "device_used_most_often",
    "genre_preference",
    "region",
    "payment_history",
]

PLAN_MAP = {
    1: "Basic",
    2: "Standard",
    3: "Premium",
}

RAW_DISPLAY_REPLACEMENTS = {
    "sci fi": "Sci-Fi",
    "on time": "On-Time",
    "smart tv": "Smart TV",
}


def load_cleaned_dataset(path: Path) -> pd.DataFrame:
    """Load the cleaned churn dataset.

    Parameters
    ----------
    path:
        Location of the cleaned CSV file.

    Returns
    -------
    pandas.DataFrame
        The cleaned dataset.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {path}. "
            "Run src/clean_data.py first."
        )
    return pd.read_csv(path)


def format_label(value: str) -> str:
    """Convert decoded category text into a readable display label.

    Parameters
    ----------
    value:
        Raw decoded category text.

    Returns
    -------
    str
        Human-readable label.
    """

    label = value.replace("_", " ").strip().lower()
    for source, target in RAW_DISPLAY_REPLACEMENTS.items():
        label = label.replace(source, target.lower())
    label = label.title()
    label = label.replace("Sci-Fi", "Sci-Fi")
    label = label.replace("On-Time", "On-Time")
    label = label.replace("Smart Tv", "Smart TV")
    return label


def decode_one_hot_group(df: pd.DataFrame, prefix: str) -> pd.Series:
    """Recover the original categorical value from a one-hot encoded group.

    Parameters
    ----------
    df:
        Cleaned dataframe containing one-hot encoded columns.
    prefix:
        Shared prefix for the encoded columns, for example
        ``device_used_most_often_``.

    Returns
    -------
    pandas.Series
        Decoded, human-readable category values.
    """

    columns = [column for column in df.columns if column.startswith(prefix)]
    if not columns:
        raise ValueError(f"No encoded columns found for prefix '{prefix}'.")

    decoded = df[columns].idxmax(axis=1).str.replace(prefix, "", regex=False)
    return decoded.map(format_label)


def reconstruct_raw_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Reconstruct the raw feature layout from the cleaned encoded dataset.

    Parameters
    ----------
    df:
        Cleaned churn dataframe produced by ``src/clean_data.py``.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.Series]
        Raw-style features and binary target series.
    """

    df = df.copy()

    required_columns = [ID_COLUMN, TARGET_COLUMN, "subscription_plan"] + NUMERIC_FEATURES
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"Expected column '{column}' was not found in the cleaned dataset.")

    raw_features = pd.DataFrame(
        {
            "subscription_length_months": df["subscription_length_months"],
            "customer_satisfaction_score": df["customer_satisfaction_score"],
            "daily_watch_time_hours": df["daily_watch_time_hours"],
            "engagement_rate": df["engagement_rate"],
            "subscription_plan": df["subscription_plan"].map(PLAN_MAP),
            "support_queries_logged": df["support_queries_logged"],
            "age": df["age"],
            "monthly_income_usd": df["monthly_income_usd"],
            "promotional_offers_used": df["promotional_offers_used"],
            "number_of_profiles_created": df["number_of_profiles_created"],
            "device_used_most_often": decode_one_hot_group(df, "device_used_most_often_"),
            "genre_preference": decode_one_hot_group(df, "genre_preference_"),
            "region": decode_one_hot_group(df, "region_"),
            "payment_history": decode_one_hot_group(df, "payment_history_"),
        }
    )

    target = df[TARGET_COLUMN].astype(int)
    if target.isna().any():
        raise ValueError("The target column contains missing values after loading.")

    return raw_features, target


def build_preprocessor() -> ColumnTransformer:
    """Build the shared preprocessing pipeline.

    Parameters
    ----------
    None

    Returns
    -------
    sklearn.compose.ColumnTransformer
        Fitted later on the training data.
    """

    numeric_transformer = Pipeline(
        steps=[
            # Scaling helps the logistic regression baseline; it is also safe
            # for the tree-based model we evaluate in parallel.
            ("scaler", StandardScaler()),
        ]
    )

    ordinal_transformer = Pipeline(
        steps=[
            ("encoder", OrdinalEncoder(
                categories=[["Basic", "Standard", "Premium"]],
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            )),
        ]
    )

    nominal_transformer = Pipeline(
        steps=[
            # One-hot encoding keeps the nominal categories separate and
            # interpretable for future explanation work.
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("ord", ordinal_transformer, ORDINAL_FEATURES),
            ("cat", nominal_transformer, NOMINAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_logistic_pipeline() -> Pipeline:
    """Create the Logistic Regression baseline pipeline.

    Parameters
    ----------
    None

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline with preprocessing and logistic regression classifier.
    """

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def build_random_forest_pipeline() -> Pipeline:
    """Create the Random Forest pipeline.

    Parameters
    ----------
    None

    Returns
    -------
    sklearn.pipeline.Pipeline
        Pipeline with preprocessing and random forest classifier.
    """

    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    random_state=42,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    min_samples_leaf=2,
                ),
            ),
        ]
    )


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Evaluate a fitted model on the held-out test set.

    Parameters
    ----------
    model:
        Fitted pipeline to evaluate.
    X_test:
        Test feature matrix.
    y_test:
        Test target values.

    Returns
    -------
    dict[str, float]
        Accuracy, precision, recall, F1, and ROC-AUC.
    """

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def run_cross_validation(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> Tuple[float, float, np.ndarray]:
    """Run stratified cross-validation using ROC-AUC scoring.

    Parameters
    ----------
    model:
        Unfitted pipeline to cross-validate.
    X:
        Full feature matrix.
    y:
        Full target vector.

    Returns
    -------
    tuple[float, float, numpy.ndarray]
        Mean ROC-AUC, standard deviation, and raw fold scores.
    """

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(scores.mean()), float(scores.std()), scores


def pretty_feature_name(name: str) -> str:
    """Make a transformed feature name easier to read.

    Parameters
    ----------
    name:
        Raw transformed feature name from ``get_feature_names_out``.

    Returns
    -------
    str
        Human-friendly feature name.
    """

    cleaned = name.replace("num__", "")
    cleaned = cleaned.replace("ord__", "")
    cleaned = cleaned.replace("cat__", "")
    cleaned = cleaned.replace("device_used_most_often_", "device_used_most_often=")
    cleaned = cleaned.replace("genre_preference_", "genre_preference=")
    cleaned = cleaned.replace("region_", "region=")
    cleaned = cleaned.replace("payment_history_", "payment_history=")
    cleaned = cleaned.replace("_", " ")
    cleaned = cleaned.replace("Customer Satisfaction Score", "Customer Satisfaction Score")
    cleaned = cleaned.replace("Daily Watch Time Hours", "Daily Watch Time Hours")
    return cleaned


def print_comparison_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Print and return a side-by-side comparison table.

    Parameters
    ----------
    results:
        Mapping of model name to metric dictionary.

    Returns
    -------
    pandas.DataFrame
        Comparison table sorted by model name.
    """

    comparison = pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1", "roc_auc"]]
    comparison = comparison.sort_index()

    print("\nTEST SET METRICS")
    print(comparison.round(4).to_string())

    return comparison


def print_feature_importances(model: Pipeline, top_n: int = 10) -> pd.DataFrame:
    """Print the top transformed feature importances from the Random Forest.

    Parameters
    ----------
    model:
        Fitted Random Forest pipeline.
    top_n:
        Number of top features to display.

    Returns
    -------
    pandas.DataFrame
        Table of the top feature importances.
    """

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    top_features = importance_df.head(top_n).copy()
    top_features["feature"] = top_features["feature"].map(pretty_feature_name)

    print("\nTOP 10 RANDOM FOREST FEATURE IMPORTANCES")
    print(top_features.to_string(index=False, formatters={"importance": "{:.4f}".format}))

    return top_features


def select_best_model(results: Dict[str, Dict[str, float]]) -> str:
    """Choose the best model, prioritizing recall over all other metrics.

    Parameters
    ----------
    results:
        Mapping of model name to metric dictionary.

    Returns
    -------
    str
        Name of the selected model.
    """

    return max(results.items(), key=lambda item: (item[1]["recall"], item[1]["roc_auc"]))[0]


def save_model_bundle(bundle: Dict[str, object], output_path: Path) -> None:
    """Persist the training artifact bundle to disk with joblib.

    Parameters
    ----------
    bundle:
        Dictionary containing the trained model and supporting preprocessing
        objects.
    output_path:
        Destination path for the serialized bundle.

    Returns
    -------
    None
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)


def main() -> None:
    """Run the full modeling workflow end to end.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    cleaned_df = load_cleaned_dataset(CLEAN_DATA_PATH)
    X, y = reconstruct_raw_features(cleaned_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    model_builders = {
        "Logistic Regression": build_logistic_pipeline,
        "Random Forest": build_random_forest_pipeline,
    }

    fitted_models: Dict[str, Pipeline] = {}
    test_results: Dict[str, Dict[str, float]] = {}

    for model_name, builder in model_builders.items():
        model = builder()
        model.fit(X_train, y_train)
        fitted_models[model_name] = model
        test_results[model_name] = evaluate_model(model, X_test, y_test)

    rf_cv_mean, rf_cv_std, rf_cv_scores = run_cross_validation(
        build_random_forest_pipeline(),
        X,
        y,
    )

    comparison = print_comparison_table(test_results)
    print(
        f"\nRANDOM FOREST 5-FOLD ROC-AUC: "
        f"{rf_cv_mean:.4f} +/- {rf_cv_std:.4f}"
    )
    print(f"Fold scores: {np.round(rf_cv_scores, 4).tolist()}")

    rf_importances = print_feature_importances(fitted_models["Random Forest"], top_n=10)

    best_model_name = select_best_model(test_results)
    best_model_template = model_builders[best_model_name]()
    best_model_template.fit(X, y)

    best_metrics = test_results[best_model_name]
    other_model_name = next(name for name in model_builders if name != best_model_name)
    other_metrics = test_results[other_model_name]

    print(
        f"\nRECOMMENDATION: {best_model_name} is the better choice for now "
        f"because it delivers the strongest recall on the holdout set "
        f"({best_metrics['recall']:.4f} vs {other_metrics['recall']:.4f}). "
        f"That is the right trade-off here because missing a churner is "
        f"costlier than flagging a false alarm."
    )

    artifact = {
        "best_model_name": best_model_name,
        "model": best_model_template,
        "preprocessor": best_model_template.named_steps["preprocessor"],
        "feature_columns": X.columns.tolist(),
        "numeric_features": NUMERIC_FEATURES,
        "ordinal_features": ORDINAL_FEATURES,
        "nominal_features": NOMINAL_FEATURES,
        "target_column": TARGET_COLUMN,
        "plan_mapping": PLAN_MAP,
        "holdout_metrics": {
            best_model_name: best_metrics,
            other_model_name: other_metrics,
        },
        "comparison_table": comparison.round(4).to_dict(orient="index"),
        "rf_cross_validation": {
            "mean_roc_auc": rf_cv_mean,
            "std_roc_auc": rf_cv_std,
            "fold_scores": rf_cv_scores.tolist(),
        },
        "rf_feature_importances": rf_importances.to_dict(orient="records"),
    }

    save_model_bundle(artifact, MODEL_OUTPUT_PATH)
    print(f"\nSaved model bundle to: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
