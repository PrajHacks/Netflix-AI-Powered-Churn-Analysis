"""Explain the Logistic Regression churn model with SHAP.

Run this script directly from the command line:

    python src/explain_model.py

What it does:
- loads the saved model bundle from ``models/churn_model.pkl``
- reconstructs the raw-style feature frame from ``outputs/cleaned_churn_data.csv``
- creates a stratified train/test split so SHAP can be applied to a held-out test set
- uses ``shap.LinearExplainer`` on the fitted Logistic Regression classifier
- saves global SHAP summary plots to ``shap_charts/``
- prints the top 5 SHAP features
- generates plain-language customer explanations for all customers
- writes ``outputs/customer_explanations.csv``
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DATA_PATH = PROJECT_ROOT / "outputs" / "cleaned_churn_data.csv"
MODEL_BUNDLE_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"
SHAP_CHARTS_DIR = PROJECT_ROOT / "shap_charts"
OUTPUT_EXPLANATIONS_PATH = PROJECT_ROOT / "outputs" / "customer_explanations.csv"

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

NUMERIC_LABELS = {
    "subscription_length_months": "Subscription length (months)",
    "customer_satisfaction_score": "Customer satisfaction score",
    "daily_watch_time_hours": "Daily watch time (hours)",
    "engagement_rate": "Engagement rate",
    "support_queries_logged": "Support queries logged",
    "age": "Age",
    "monthly_income_usd": "Monthly income (USD)",
    "promotional_offers_used": "Promotional offers used",
    "number_of_profiles_created": "Number of profiles created",
}

ORDINAL_LABELS = {
    "subscription_plan": "Subscription plan",
}

NOMINAL_LABELS = {
    "device_used_most_often": "Device used most often",
    "genre_preference": "Genre preference",
    "region": "Region",
    "payment_history": "Payment history",
}

MISSING_CATEGORY_TOKEN = "Missing"


@dataclass
class FeatureGroup:
    """Metadata for a transformed feature column.

    Attributes
    ----------
    transformed_name:
        Exact transformed feature name from the fitted preprocessing pipeline.
    group_name:
        Raw feature group name used for aggregation.
    display_name:
        Human-readable label for charts and outputs.
    kind:
        One of ``numeric``, ``ordinal``, or ``categorical``.
    category_value:
        The active category value for one-hot encoded features.
    index:
        Column index in the transformed feature matrix.
    """

    transformed_name: str
    group_name: str
    display_name: str
    kind: str
    category_value: Optional[str]
    index: int


@dataclass
class ExplanationContext:
    """Runtime objects needed to generate explanations."""

    model: object
    preprocessor: object
    classifier: object
    feature_names: List[str]
    feature_labels: List[str]
    feature_groups: List[FeatureGroup]
    group_to_indices: Dict[str, List[int]]
    customer_ids: pd.Series
    raw_features: pd.DataFrame
    target: pd.Series
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    Xt_train: np.ndarray
    Xt_test: np.ndarray
    Xt_all: np.ndarray
    shap_values_test: np.ndarray
    shap_values_all: np.ndarray
    mean_abs_shap: pd.Series
    customer_id_to_index: Dict[str, int]


_CONTEXT: Optional[ExplanationContext] = None


def load_cleaned_dataset(path: Path) -> pd.DataFrame:
    """Load the cleaned churn dataset from disk.

    Parameters
    ----------
    path:
        Path to ``outputs/cleaned_churn_data.csv``.

    Returns
    -------
    pandas.DataFrame
        The cleaned and encoded churn dataset.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {path}. Run src/clean_data.py first."
        )
    return pd.read_csv(path)


def load_model_bundle(path: Path) -> dict:
    """Load the serialized model bundle.

    Parameters
    ----------
    path:
        Path to ``models/churn_model.pkl``.

    Returns
    -------
    dict
        The joblib bundle containing the fitted pipeline and metadata.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Model bundle not found at {path}. Run src/train_model.py first."
        )
    return joblib.load(path)


def humanize_category(value: str) -> str:
    """Convert an encoded category label into a polished display string.

    Parameters
    ----------
    value:
        Raw category text from the encoded feature name.

    Returns
    -------
    str
        Human-readable category string.
    """

    text = value.replace("_", " ").strip()
    lower = text.lower()
    if lower == "sci fi":
        return "Sci-Fi"
    if lower == "on time":
        return "On-Time"
    if lower == "smart tv":
        return "Smart TV"
    return text.title()


def format_label(value: str) -> str:
    """Convert decoded category text into a readable label.

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
    label = label.replace("sci fi", "sci-fi")
    label = label.replace("on time", "on-time")
    label = label.replace("smart tv", "smart tv")
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
        Shared prefix for the encoded columns.

    Returns
    -------
    pandas.Series
        Decoded human-readable category values.
    """

    columns = [column for column in df.columns if column.startswith(prefix)]
    if not columns:
        raise ValueError(f"No encoded columns found for prefix '{prefix}'.")

    decoded = df[columns].idxmax(axis=1).str.replace(prefix, "", regex=False)
    return decoded.map(format_label)


def reconstruct_raw_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Reconstruct the raw feature frame from the cleaned encoded dataset.

    Parameters
    ----------
    df:
        Cleaned churn dataframe created by ``src/clean_data.py``.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.Series, pandas.Series]
        Raw-style features, binary target, and customer IDs aligned by row.
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
    customer_ids = df[ID_COLUMN].astype(str)

    if raw_features.isna().any().any():
        raise ValueError("Unexpected missing values while reconstructing raw features.")

    return raw_features, target, customer_ids


def build_feature_metadata(feature_names: List[str]) -> Tuple[List[FeatureGroup], Dict[str, List[int]], List[str]]:
    """Build metadata and grouping information for transformed features.

    Parameters
    ----------
    feature_names:
        Names returned by ``preprocessor.get_feature_names_out()``.

    Returns
    -------
    tuple[list[FeatureGroup], dict[str, list[int]], list[str]]
        Per-column metadata, grouping map, and display labels.
    """

    groups: List[FeatureGroup] = []
    group_to_indices: Dict[str, List[int]] = defaultdict(list)
    pretty_labels: List[str] = []

    for idx, transformed_name in enumerate(feature_names):
        if "__" not in transformed_name:
            raise ValueError(f"Unexpected transformed feature name: {transformed_name}")

        prefix, remainder = transformed_name.split("__", 1)
        if prefix == "num":
            group_name = remainder
            display_name = NUMERIC_LABELS.get(group_name, group_name.replace("_", " ").title())
            kind = "numeric"
            category_value = None
        elif prefix == "ord":
            group_name = remainder
            display_name = ORDINAL_LABELS.get(group_name, group_name.replace("_", " ").title())
            kind = "ordinal"
            category_value = None
        elif prefix == "cat":
            group_name = None
            category_value = None
            for nominal_group in NOMINAL_FEATURES:
                nominal_prefix = f"{nominal_group}_"
                if remainder.startswith(nominal_prefix):
                    group_name = nominal_group
                    category_value = remainder[len(nominal_prefix) :]
                    display_name = f"{NOMINAL_LABELS[nominal_group]}: {humanize_category(category_value)}"
                    kind = "categorical"
                    break
            if group_name is None:
                raise ValueError(f"Could not parse categorical transformed feature '{transformed_name}'.")
        else:
            raise ValueError(f"Unrecognized transformed feature prefix in '{transformed_name}'.")

        groups.append(
            FeatureGroup(
                transformed_name=transformed_name,
                group_name=group_name,
                display_name=display_name,
                kind=kind,
                category_value=category_value,
                index=idx,
            )
        )
        group_to_indices[group_name].append(idx)
        pretty_labels.append(display_name)

    return groups, group_to_indices, pretty_labels


def build_explanation_context() -> ExplanationContext:
    """Load the cleaned data and model bundle, then prepare SHAP inputs.

    Parameters
    ----------
    None

    Returns
    -------
    ExplanationContext
        Fully prepared runtime context for SHAP explanations.
    """

    cleaned_df = load_cleaned_dataset(CLEAN_DATA_PATH)
    bundle = load_model_bundle(MODEL_BUNDLE_PATH)

    if bundle.get("best_model_name") != "Logistic Regression":
        print(
            "Warning: the saved bundle is not marked as Logistic Regression. "
            "Proceeding anyway because the script is designed for a linear model."
        )

    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    classifier = model.named_steps["classifier"]

    raw_features, target, customer_ids = reconstruct_raw_features(cleaned_df)

    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        raw_features,
        target,
        customer_ids,
        test_size=0.2,
        stratify=target,
        random_state=42,
    )

    Xt_train = preprocessor.transform(X_train)
    Xt_test = preprocessor.transform(X_test)
    Xt_all = preprocessor.transform(raw_features)

    feature_names = list(preprocessor.get_feature_names_out())
    feature_groups, group_to_indices, feature_labels = build_feature_metadata(feature_names)

    # SHAP is applied to the linear classifier on the transformed feature matrix.
    explainer = shap.LinearExplainer(classifier, Xt_train)
    shap_values_test = explainer.shap_values(Xt_test)
    shap_values_all = explainer.shap_values(Xt_all)

    mean_abs_shap = pd.Series(
        np.abs(shap_values_test).mean(axis=0),
        index=feature_labels,
    ).sort_values(ascending=False)

    customer_id_to_index = {customer_id: idx for idx, customer_id in enumerate(customer_ids.tolist())}

    return ExplanationContext(
        model=model,
        preprocessor=preprocessor,
        classifier=classifier,
        feature_names=feature_names,
        feature_labels=feature_labels,
        feature_groups=feature_groups,
        group_to_indices=group_to_indices,
        customer_ids=customer_ids,
        raw_features=raw_features,
        target=target,
        X_train=X_train,
        X_test=X_test,
        Xt_train=Xt_train,
        Xt_test=Xt_test,
        Xt_all=Xt_all,
        shap_values_test=shap_values_test,
        shap_values_all=shap_values_all,
        mean_abs_shap=mean_abs_shap,
        customer_id_to_index=customer_id_to_index,
    )


def ensure_output_directories() -> None:
    """Create the output folders used by this script.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    SHAP_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_EXPLANATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)


def save_shap_plots(context: ExplanationContext) -> None:
    """Save the global SHAP summary plots.

    Parameters
    ----------
    context:
        Prepared explanation context with SHAP values for the test set.

    Returns
    -------
    None
    """

    max_display = min(20, len(context.feature_labels))

    transformed_test_frame = pd.DataFrame(
        context.Xt_test,
        columns=context.feature_labels,
        index=context.X_test.index,
    )

    plt.figure()
    shap.summary_plot(
        context.shap_values_test,
        transformed_test_frame,
        feature_names=context.feature_labels,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(SHAP_CHARTS_DIR / "01_shap_summary_beeswarm.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(
        context.shap_values_test,
        transformed_test_frame,
        feature_names=context.feature_labels,
        plot_type="bar",
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(SHAP_CHARTS_DIR / "02_shap_summary_bar.png", bbox_inches="tight")
    plt.close()


def describe_numeric_reason(feature_name: str, value: float, contribution: float) -> str:
    """Translate a numeric feature and SHAP contribution into plain language.

    Parameters
    ----------
    feature_name:
        Raw feature name.
    value:
        The customer's observed value.
    contribution:
        Grouped SHAP contribution for the feature.

    Returns
    -------
    str
        Business-friendly explanation sentence.
    """

    direction = "increased" if contribution > 0 else "decreased"
    if feature_name == "customer_satisfaction_score":
        if value <= 4:
            prefix = f"Low satisfaction score ({int(value)}/10)"
        elif value >= 7:
            prefix = f"High satisfaction score ({int(value)}/10)"
        else:
            prefix = f"Satisfaction score ({int(value)}/10)"
    elif feature_name == "engagement_rate":
        if value <= 4:
            prefix = f"Low engagement rate ({int(value)}/10)"
        elif value >= 7:
            prefix = f"High engagement rate ({int(value)}/10)"
        else:
            prefix = f"Engagement rate ({int(value)}/10)"
    elif feature_name == "subscription_length_months":
        if value <= 6:
            prefix = f"Short subscription history ({int(value)} months)"
        elif value >= 12:
            prefix = f"Long subscription history ({int(value)} months)"
        else:
            prefix = f"Subscription length ({int(value)} months)"
    elif feature_name == "support_queries_logged":
        if value <= 2:
            prefix = f"Few support queries ({int(value)})"
        elif value >= 7:
            prefix = f"Frequent support queries ({int(value)})"
        else:
            prefix = f"Support queries logged ({int(value)})"
    elif feature_name == "promotional_offers_used":
        if value <= 1:
            prefix = f"Few promotional offers used ({int(value)})"
        elif value >= 4:
            prefix = f"Many promotional offers used ({int(value)})"
        else:
            prefix = f"Promotional offers used ({int(value)})"
    elif feature_name == "daily_watch_time_hours":
        if value <= 1.5:
            prefix = f"Low daily watch time ({value:.1f} hours)"
        elif value >= 3.5:
            prefix = f"High daily watch time ({value:.1f} hours)"
        else:
            prefix = f"Daily watch time ({value:.1f} hours)"
    elif feature_name == "age":
        if value <= 30:
            prefix = f"Younger age ({int(value)})"
        elif value >= 55:
            prefix = f"Older age ({int(value)})"
        else:
            prefix = f"Age ({int(value)})"
    elif feature_name == "monthly_income_usd":
        if value <= 3500:
            prefix = f"Lower monthly income (${int(value):,})"
        elif value >= 7500:
            prefix = f"Higher monthly income (${int(value):,})"
        else:
            prefix = f"Monthly income (${int(value):,})"
    elif feature_name == "number_of_profiles_created":
        if value <= 2:
            prefix = f"Few profiles created ({int(value)})"
        elif value >= 4:
            prefix = f"Many profiles created ({int(value)})"
        else:
            prefix = f"Profiles created ({int(value)})"
    else:
        prefix = f"{NUMERIC_LABELS.get(feature_name, feature_name.replace('_', ' ').title())} ({value})"

    return f"{prefix} {direction} churn risk"


def describe_categorical_reason(feature_name: str, value: str, contribution: float) -> str:
    """Translate a categorical feature and SHAP contribution into plain language.

    Parameters
    ----------
    feature_name:
        Raw categorical feature name.
    value:
        The active category for the customer.
    contribution:
        Grouped SHAP contribution for the feature.

    Returns
    -------
    str
        Business-friendly explanation sentence.
    """

    direction = "increased" if contribution > 0 else "decreased"

    if feature_name == "subscription_plan":
        prefix = f"{value} plan"
    elif feature_name == "payment_history":
        prefix = f"{value} payment history"
    elif feature_name == "device_used_most_often":
        prefix = f"{value} device usage"
    elif feature_name == "genre_preference":
        prefix = f"{value} genre preference"
    elif feature_name == "region":
        prefix = f"Being in {value}"
    else:
        prefix = f"{NOMINAL_LABELS.get(feature_name, feature_name.replace('_', ' ').title())}: {value}"

    return f"{prefix} {direction} churn risk"


def build_reason_sentence(raw_row: pd.Series, group_name: str, contribution: float) -> str:
    """Build a plain-language reason sentence for one grouped SHAP contribution.

    Parameters
    ----------
    raw_row:
        One customer row from the reconstructed raw feature frame.
    group_name:
        Raw feature group name.
    contribution:
        Grouped SHAP contribution.

    Returns
    -------
    str
        Plain-language reason sentence.
    """

    value = raw_row[group_name]
    if group_name in NUMERIC_FEATURES:
        return describe_numeric_reason(group_name, float(value), contribution)
    return describe_categorical_reason(group_name, str(value), contribution)


def aggregate_group_contributions(shap_row: np.ndarray, context: ExplanationContext) -> Dict[str, float]:
    """Aggregate transformed SHAP values back to raw feature groups.

    Parameters
    ----------
    shap_row:
        A single row of SHAP values from the transformed feature matrix.
    context:
        Prepared explanation context.

    Returns
    -------
    dict[str, float]
        Mapping from raw feature group name to grouped SHAP contribution.
    """

    return {
        group_name: float(shap_row[index_list].sum())
        for group_name, index_list in context.group_to_indices.items()
    }


def explain_customer(customer_id: str) -> Dict[str, object]:
    """Explain churn risk for a single customer.

    Parameters
    ----------
    customer_id:
        Customer identifier from the cleaned dataset.

    Returns
    -------
    dict[str, object]
        Churn probability and the top 3 SHAP-driven reasons in plain language.
    """

    if _CONTEXT is None:
        raise RuntimeError("Explainability context is not initialized. Run main() first.")

    if customer_id not in _CONTEXT.customer_id_to_index:
        raise KeyError(f"Customer ID '{customer_id}' was not found in the dataset.")

    idx = _CONTEXT.customer_id_to_index[customer_id]
    raw_row = _CONTEXT.raw_features.iloc[idx]
    shap_row = _CONTEXT.shap_values_all[idx]
    grouped = aggregate_group_contributions(shap_row, _CONTEXT)
    top_groups = sorted(grouped.items(), key=lambda item: abs(item[1]), reverse=True)[:3]
    reasons = [build_reason_sentence(raw_row, group_name, contribution) for group_name, contribution in top_groups]
    probability = float(_CONTEXT.model.predict_proba(_CONTEXT.raw_features.iloc[[idx]])[0, 1])

    return {
        "customer_id": customer_id,
        "churn_probability": probability,
        "reason_1": reasons[0] if len(reasons) > 0 else "",
        "reason_2": reasons[1] if len(reasons) > 1 else "",
        "reason_3": reasons[2] if len(reasons) > 2 else "",
    }


def build_all_customer_explanations(context: ExplanationContext) -> pd.DataFrame:
    """Generate customer-level explanations for the full customer base.

    Parameters
    ----------
    context:
        Prepared explanation context.

    Returns
    -------
    pandas.DataFrame
        Table with customer IDs, churn probabilities, and top 3 reasons.
    """

    records: List[Dict[str, object]] = []
    for customer_id in context.customer_ids.tolist():
        record = explain_customer(customer_id)
        records.append(record)
    return pd.DataFrame(records)


def select_sample_customer_ids(context: ExplanationContext, explanations: pd.DataFrame) -> Dict[str, str]:
    """Select representative high-risk, low-risk, and borderline customers.

    Parameters
    ----------
    context:
        Prepared explanation context.
    explanations:
        Customer explanations table containing churn probabilities.

    Returns
    -------
    dict[str, str]
        Mapping of sample label to customer ID.
    """

    high_risk_idx = explanations["churn_probability"].idxmax()
    low_risk_idx = explanations["churn_probability"].idxmin()
    borderline_idx = (explanations["churn_probability"] - 0.5).abs().idxmin()

    return {
        "high_risk": str(explanations.loc[high_risk_idx, "customer_id"]),
        "low_risk": str(explanations.loc[low_risk_idx, "customer_id"]),
        "borderline": str(explanations.loc[borderline_idx, "customer_id"]),
    }


def print_top_global_features(context: ExplanationContext) -> None:
    """Print the top 5 features by mean absolute SHAP value.

    Parameters
    ----------
    context:
        Prepared explanation context.

    Returns
    -------
    None
    """

    print("\nTOP 5 GLOBAL FEATURES BY MEAN |SHAP|")
    print(
        context.mean_abs_shap.head(5)
        .rename_axis("feature")
        .reset_index(name="mean_abs_shap")
        .to_string(index=False, formatters={"mean_abs_shap": "{:.4f}".format})
    )


def print_sample_explanations(explanations: pd.DataFrame, sample_ids: Dict[str, str]) -> None:
    """Print three representative customer explanations.

    Parameters
    ----------
    explanations:
        Customer explanations table.
    sample_ids:
        Mapping from sample label to customer ID.

    Returns
    -------
    None
    """

    label_titles = {
        "high_risk": "High-risk",
        "low_risk": "Low-risk",
        "borderline": "Borderline",
    }

    print("\nSAMPLE CUSTOMER EXPLANATIONS")
    for label, customer_id in sample_ids.items():
        row = explanations.loc[explanations["customer_id"] == customer_id].iloc[0]
        print(f"\n{label_titles[label]} customer: {customer_id}")
        print(f"Churn probability: {row['churn_probability']:.4f}")
        print(f"1. {row['reason_1']}")
        print(f"2. {row['reason_2']}")
        print(f"3. {row['reason_3']}")


def save_customer_explanations(context: ExplanationContext, explanations: pd.DataFrame) -> None:
    """Save the full customer explanation export.

    Parameters
    ----------
    context:
        Prepared explanation context.
    explanations:
        Customer explanations table.

    Returns
    -------
    None
    """

    output = explanations[["customer_id", "churn_probability", "reason_1", "reason_2", "reason_3"]].copy()
    output.to_csv(OUTPUT_EXPLANATIONS_PATH, index=False)
    print(f"\nSaved customer explanations to: {OUTPUT_EXPLANATIONS_PATH}")


def main() -> None:
    """Run the SHAP explainability workflow end to end.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    global _CONTEXT

    ensure_output_directories()
    context = build_explanation_context()
    _CONTEXT = context

    save_shap_plots(context)

    print_top_global_features(context)

    explanations = build_all_customer_explanations(context)
    save_customer_explanations(context, explanations)

    sample_ids = select_sample_customer_ids(context, explanations)
    print_sample_explanations(explanations, sample_ids)


if __name__ == "__main__":
    main()
