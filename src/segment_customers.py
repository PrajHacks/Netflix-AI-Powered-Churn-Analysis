"""Segment customers with K-Means and build the master analytics dataset.

Run this script directly from the command line:

    python src/segment_customers.py

What it does:
- loads ``outputs/cleaned_churn_data.csv`` and ``outputs/customer_explanations.csv``
- selects satisfaction, engagement, and encoded payment history as clustering features
- scales the clustering features with ``StandardScaler``
- evaluates K-Means for k=2 through k=6 using inertia and silhouette score
- saves elbow and silhouette charts to ``segmentation_charts/``
- chooses an interpretable cluster count from the data
- fits K-Means and profiles each cluster
- generates business-friendly cluster names from the actual cluster statistics
- saves a scatter plot of the clustered customers
- writes ``outputs/final_customer_dataset.csv`` for Power BI and the backend
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DATA_PATH = PROJECT_ROOT / "outputs" / "cleaned_churn_data.csv"
EXPLANATIONS_PATH = PROJECT_ROOT / "outputs" / "customer_explanations.csv"
SEGMENTATION_CHARTS_DIR = PROJECT_ROOT / "segmentation_charts"
FINAL_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "final_customer_dataset.csv"
SEGMENT_PROFILE_PATH = PROJECT_ROOT / "outputs" / "cluster_profiles.csv"
KMEANS_SELECTION_PATH = PROJECT_ROOT / "outputs" / "kmeans_selection_metrics.csv"

ID_COLUMN = "customer_id"
TARGET_COLUMN = "churn_status"

SEGMENTATION_FEATURES = [
    "customer_satisfaction_score",
    "engagement_rate",
    "payment_history_delayed",
]

K_RANGE = range(2, 7)
RANDOM_STATE = 42


def load_csv(path: Path, description: str) -> pd.DataFrame:
    """Load a CSV file from disk.

    Parameters
    ----------
    path:
        File path to load.
    description:
        Friendly dataset name used in error messages.

    Returns
    -------
    pandas.DataFrame
        Loaded dataframe.
    """

    if not path.exists():
        raise FileNotFoundError(f"{description} not found at {path}.")
    return pd.read_csv(path)


def ensure_required_columns(df: pd.DataFrame, required_columns: List[str], dataset_name: str) -> None:
    """Verify that the dataframe contains the expected columns.

    Parameters
    ----------
    df:
        Dataframe to validate.
    required_columns:
        Columns that must be present.
    dataset_name:
        Name of the dataset for readable errors.

    Returns
    -------
    None
    """

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")


def get_payment_history_feature(df: pd.DataFrame) -> pd.Series:
    """Return a single encoded payment-history feature for clustering.

    Parameters
    ----------
    df:
        Cleaned customer dataframe.

    Returns
    -------
    pandas.Series
        Binary payment-history feature where 1 means delayed payment.
    """

    if "payment_history_delayed" in df.columns:
        return df["payment_history_delayed"].astype(float)
    if "payment_history_on_time" in df.columns:
        # Use the delayed indicator rather than both one-hot columns so the
        # clustering signal stays compact and avoids redundant duplication.
        return (1 - df["payment_history_on_time"].astype(float)).astype(float)
    raise ValueError(
        "Neither 'payment_history_delayed' nor 'payment_history_on_time' was found "
        "in the cleaned dataset."
    )


def build_segmentation_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build the feature matrix used for customer segmentation.

    Parameters
    ----------
    df:
        Cleaned customer dataframe.

    Returns
    -------
    pandas.DataFrame
        Segmentation feature frame with satisfaction, engagement, and delayed-payment encoding.
    """

    ensure_required_columns(
        df,
        ["customer_satisfaction_score", "engagement_rate"],
        "Cleaned customer dataset",
    )

    segmentation_frame = pd.DataFrame(
        {
            "customer_satisfaction_score": df["customer_satisfaction_score"].astype(float),
            "engagement_rate": df["engagement_rate"].astype(float),
            "payment_history_delayed": get_payment_history_feature(df),
        }
    )

    return segmentation_frame


def evaluate_kmeans_range(scaled_features: np.ndarray, k_values: range) -> pd.DataFrame:
    """Fit K-Means for several k values and collect model-selection metrics.

    Parameters
    ----------
    scaled_features:
        Standardized clustering feature matrix.
    k_values:
        Range of cluster counts to evaluate.

    Returns
    -------
    pandas.DataFrame
        DataFrame with k, inertia, and silhouette score for each candidate.
    """

    records: List[Dict[str, float]] = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
        labels = model.fit_predict(scaled_features)
        records.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette_score": float(silhouette_score(scaled_features, labels)),
            }
        )

    metrics = pd.DataFrame(records)
    metrics.to_csv(KMEANS_SELECTION_PATH, index=False)
    return metrics


def compute_elbow_k(metrics_df: pd.DataFrame) -> int:
    """Compute the elbow point using maximum distance from the inertia line.

    Parameters
    ----------
    metrics_df:
        K-selection table containing k and inertia.

    Returns
    -------
    int
        Elbow-based cluster count.
    """

    ks = metrics_df["k"].to_numpy(dtype=float)
    inertia = metrics_df["inertia"].to_numpy(dtype=float)

    x1, y1 = ks[0], inertia[0]
    x2, y2 = ks[-1], inertia[-1]
    numerator = np.abs((y2 - y1) * ks - (x2 - x1) * inertia + x2 * y1 - y2 * x1)
    denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    distances = numerator / denominator
    return int(ks[np.argmax(distances)])


def choose_best_k(metrics_df: pd.DataFrame, silhouette_tolerance: float = 0.02) -> Tuple[int, str]:
    """Choose a cluster count using the elbow point and silhouette score.

    Parameters
    ----------
    metrics_df:
        K-selection table.
    silhouette_tolerance:
        Maximum silhouette gap allowed between the elbow choice and the best score.

    Returns
    -------
    tuple[int, str]
        Chosen k and a short explanation.
    """

    elbow_k = compute_elbow_k(metrics_df)
    best_silhouette_row = metrics_df.loc[metrics_df["silhouette_score"].idxmax()]
    best_silhouette_k = int(best_silhouette_row["k"])
    best_silhouette_score = float(best_silhouette_row["silhouette_score"])

    elbow_score = float(metrics_df.loc[metrics_df["k"] == elbow_k, "silhouette_score"].iloc[0])
    silhouette_gap = best_silhouette_score - elbow_score

    if silhouette_gap <= silhouette_tolerance:
        chosen_k = elbow_k
        reason = (
            f"k={chosen_k} is the elbow point in the inertia curve and its silhouette "
            f"score ({elbow_score:.4f}) is within {silhouette_gap:.4f} of the best score "
            f"({best_silhouette_score:.4f} at k={best_silhouette_k}), so it keeps the "
            f"segments interpretable without giving up much cluster quality."
        )
    else:
        chosen_k = best_silhouette_k
        reason = (
            f"k={chosen_k} maximizes silhouette score ({best_silhouette_score:.4f}); "
            f"the elbow choice ({elbow_k}) would lose too much separation."
        )

    return chosen_k, reason


def save_k_selection_charts(metrics_df: pd.DataFrame) -> None:
    """Save elbow and silhouette charts for the K-Means selection step.

    Parameters
    ----------
    metrics_df:
        K-selection table.

    Returns
    -------
    None
    """

    SEGMENTATION_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    elbow_fig, elbow_ax = plt.subplots(figsize=(8, 5))
    elbow_ax.plot(metrics_df["k"], metrics_df["inertia"], marker="o", linewidth=2, color="#1f77b4")
    elbow_ax.set_title("K-Means Elbow Curve")
    elbow_ax.set_xlabel("Number of clusters (k)")
    elbow_ax.set_ylabel("Inertia")
    elbow_ax.set_xticks(metrics_df["k"].tolist())
    elbow_ax.annotate(
        "Lower inertia is better,\nthen the curve flattens",
        xy=(metrics_df.iloc[-1]["k"], metrics_df.iloc[-1]["inertia"]),
        xytext=(0.62, 0.85),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="#555555"),
        fontsize=9,
    )
    elbow_fig.tight_layout()
    elbow_fig.savefig(SEGMENTATION_CHARTS_DIR / "01_kmeans_elbow_curve.png", dpi=160, bbox_inches="tight")
    plt.close(elbow_fig)

    sil_fig, sil_ax = plt.subplots(figsize=(8, 5))
    sil_ax.plot(
        metrics_df["k"],
        metrics_df["silhouette_score"],
        marker="o",
        linewidth=2,
        color="#d62728",
    )
    sil_ax.set_title("K-Means Silhouette Scores")
    sil_ax.set_xlabel("Number of clusters (k)")
    sil_ax.set_ylabel("Silhouette score")
    sil_ax.set_xticks(metrics_df["k"].tolist())
    sil_ax.annotate(
        "Higher is better",
        xy=(metrics_df.loc[metrics_df["silhouette_score"].idxmax(), "k"], metrics_df["silhouette_score"].max()),
        xytext=(0.66, 0.13),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="#555555"),
        fontsize=9,
    )
    sil_fig.tight_layout()
    sil_fig.savefig(SEGMENTATION_CHARTS_DIR / "02_kmeans_silhouette_scores.png", dpi=160, bbox_inches="tight")
    plt.close(sil_fig)


def build_global_reference(cleaned_df: pd.DataFrame, explanations_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate global statistics used to name and interpret clusters.

    Parameters
    ----------
    cleaned_df:
        Cleaned customer dataframe.
    explanations_df:
        Customer explanation export with churn probabilities.

    Returns
    -------
    dict[str, float]
        Summary statistics for the full customer base.
    """

    return {
        "satisfaction_q1": float(cleaned_df["customer_satisfaction_score"].quantile(0.25)),
        "satisfaction_q3": float(cleaned_df["customer_satisfaction_score"].quantile(0.75)),
        "engagement_q1": float(cleaned_df["engagement_rate"].quantile(0.25)),
        "engagement_q3": float(cleaned_df["engagement_rate"].quantile(0.75)),
        "subscription_q1": float(cleaned_df["subscription_length_months"].quantile(0.25)),
        "subscription_q3": float(cleaned_df["subscription_length_months"].quantile(0.75)),
        "income_q1": float(cleaned_df["monthly_income_usd"].quantile(0.25)),
        "income_q3": float(cleaned_df["monthly_income_usd"].quantile(0.75)),
        "avg_churn_probability": float(explanations_df["churn_probability"].mean()),
        "delayed_rate": float(cleaned_df["payment_history_delayed"].mean()),
    }


def generate_segment_name(profile_row: pd.Series, global_reference: Dict[str, float]) -> str:
    """Create a business-friendly segment name from cluster profile statistics.

    Parameters
    ----------
    profile_row:
        One row from the cluster profile table.
    global_reference:
        Global dataset statistics used to decide thresholds.

    Returns
    -------
    str
        Human-readable cluster label.
    """

    avg_churn = float(profile_row["avg_churn_probability"])
    avg_satisfaction = float(profile_row["avg_satisfaction_score"])
    avg_engagement = float(profile_row["avg_engagement_rate"])
    delayed_share = float(profile_row["pct_delayed_payments"])
    avg_income = float(profile_row["avg_monthly_income_usd"])
    avg_subscription_length = float(profile_row["avg_subscription_length_months"])

    high_churn = avg_churn >= 0.50
    low_churn = avg_churn <= min(0.35, global_reference["avg_churn_probability"] - 0.06)
    low_satisfaction = avg_satisfaction <= global_reference["satisfaction_q1"] + 0.6
    high_satisfaction = avg_satisfaction >= global_reference["satisfaction_q3"] - 0.6
    low_engagement = avg_engagement <= global_reference["engagement_q1"] + 1.0
    high_engagement = avg_engagement >= global_reference["engagement_q3"] - 1.0
    delayed_heavy = delayed_share >= max(0.60, global_reference["delayed_rate"] + 0.10)
    low_income = avg_income <= global_reference["income_q1"]
    short_history = avg_subscription_length <= global_reference["subscription_q1"] + 1.0

    # The naming is rule-based, but the rules are driven by the actual cluster
    # statistics so the segment label reflects the observed customer profile.
    if high_churn and low_engagement and delayed_heavy:
        return "High-Risk Disengaged"
    if high_churn and low_satisfaction:
        return "Low-Satisfaction At-Risk"
    if high_churn and delayed_heavy and low_income:
        return "Price-Sensitive At-Risk"
    if low_churn and high_satisfaction and high_engagement:
        return "Loyal High-Engagement"
    if low_churn and high_satisfaction:
        return "Satisfied Loyal"
    if high_engagement and delayed_heavy:
        return "Engaged but Payment-Fragile"
    if low_satisfaction:
        return "Low-Satisfaction At-Risk" if high_churn else "Low-Satisfaction"
    if short_history and avg_churn <= global_reference["avg_churn_probability"]:
        return "New/Undecided"
    if high_churn and delayed_heavy:
        return "Payment-Fragile At-Risk"
    if high_churn:
        return "At-Risk Mixed-Engagement"
    if low_churn:
        return "Stable Loyal"
    return "Mixed-Risk Mixed-Engagement"


def profile_clusters(clustered_df: pd.DataFrame, global_reference: Dict[str, float]) -> pd.DataFrame:
    """Profile each cluster and assign a business-friendly segment name.

    Parameters
    ----------
    clustered_df:
        Customer dataframe with a ``cluster_id`` column and churn explanations merged in.
    global_reference:
        Global statistics used to create segment labels.

    Returns
    -------
    pandas.DataFrame
        Cluster profile table with segment names and business metrics.
    """

    total_customers = len(clustered_df)
    profiles = (
        clustered_df.groupby("cluster_id")
        .agg(
            size=(ID_COLUMN, "size"),
            avg_churn_probability=("churn_probability", "mean"),
            avg_satisfaction_score=("customer_satisfaction_score", "mean"),
            avg_engagement_rate=("engagement_rate", "mean"),
            pct_delayed_payments=("payment_history_delayed", "mean"),
            avg_subscription_length_months=("subscription_length_months", "mean"),
            avg_monthly_income_usd=("monthly_income_usd", "mean"),
        )
        .reset_index()
    )

    profiles["pct_of_total"] = profiles["size"] / total_customers
    profiles["segment_name"] = profiles.apply(
        lambda row: generate_segment_name(row, global_reference),
        axis=1,
    )

    profiles = profiles[
        [
            "cluster_id",
            "segment_name",
            "size",
            "pct_of_total",
            "avg_churn_probability",
            "avg_satisfaction_score",
            "avg_engagement_rate",
            "pct_delayed_payments",
            "avg_subscription_length_months",
            "avg_monthly_income_usd",
        ]
    ].sort_values("cluster_id").reset_index(drop=True)

    profiles.to_csv(SEGMENT_PROFILE_PATH, index=False)
    return profiles


def print_cluster_profiles(profiles_df: pd.DataFrame) -> None:
    """Print the cluster profile table in a business-friendly format.

    Parameters
    ----------
    profiles_df:
        Cluster profile dataframe.

    Returns
    -------
    None
    """

    display_df = profiles_df.copy()
    display_df["pct_of_total"] = (display_df["pct_of_total"] * 100).round(1)
    display_df["avg_churn_probability"] = display_df["avg_churn_probability"].round(3)
    display_df["avg_satisfaction_score"] = display_df["avg_satisfaction_score"].round(2)
    display_df["avg_engagement_rate"] = display_df["avg_engagement_rate"].round(2)
    display_df["pct_delayed_payments"] = (display_df["pct_delayed_payments"] * 100).round(1)
    display_df["avg_subscription_length_months"] = display_df["avg_subscription_length_months"].round(2)
    display_df["avg_monthly_income_usd"] = display_df["avg_monthly_income_usd"].round(0).astype(int)

    print("\nCLUSTER PROFILES")
    print(
        display_df.rename(
            columns={
                "cluster_id": "cluster",
                "pct_of_total": "pct_of_total",
                "avg_churn_probability": "avg_churn_probability",
                "avg_satisfaction_score": "avg_satisfaction_score",
                "avg_engagement_rate": "avg_engagement_rate",
                "pct_delayed_payments": "pct_delayed_payments",
                "avg_subscription_length_months": "avg_subscription_length_months",
                "avg_monthly_income_usd": "avg_monthly_income_usd",
            }
        ).to_string(index=False)
    )


def save_cluster_scatter_plot(
    clustered_df: pd.DataFrame,
    scaler: StandardScaler,
    kmeans: KMeans,
    profiles_df: pd.DataFrame,
) -> None:
    """Create a 2D scatter plot of customers colored by segment.

    Parameters
    ----------
    clustered_df:
        Customer dataframe containing cluster assignments and segment names.
    scaler:
        Fitted scaler used to standardize the clustering features.
    kmeans:
        Fitted K-Means model.
    profiles_df:
        Cluster profile table with segment names.

    Returns
    -------
    None
    """

    SEGMENTATION_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    plot_df = clustered_df[[ID_COLUMN, "cluster_id", "segment_name", "customer_satisfaction_score", "engagement_rate"]].copy()
    plot_df["segment_name"] = pd.Categorical(
        plot_df["segment_name"],
        categories=profiles_df.sort_values("cluster_id")["segment_name"].tolist(),
        ordered=True,
    )

    centroids_raw = scaler.inverse_transform(kmeans.cluster_centers_)

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = sns.scatterplot(
        data=plot_df,
        x="customer_satisfaction_score",
        y="engagement_rate",
        hue="segment_name",
        palette="tab10",
        s=55,
        alpha=0.8,
        ax=ax,
    )

    ax.scatter(
        centroids_raw[:, 0],
        centroids_raw[:, 1],
        c="black",
        marker="X",
        s=180,
        label="Cluster centers",
        linewidths=1.5,
    )

    ax.set_title("Customer Segments by Satisfaction and Engagement")
    ax.set_xlabel("Customer Satisfaction Score")
    ax.set_ylabel("Engagement Rate")
    ax.legend(title="Segment", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(SEGMENTATION_CHARTS_DIR / "03_customer_segments_scatter.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_final_dataset(
    cleaned_df: pd.DataFrame,
    explanations_df: pd.DataFrame,
    clustered_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the final dashboard-ready customer dataset.

    Parameters
    ----------
    cleaned_df:
        Original cleaned customer dataframe.
    explanations_df:
        Customer explanation export.
    clustered_df:
        Cleaned customer dataframe with cluster_id and segment_name assigned.

    Returns
    -------
    pandas.DataFrame
        Final merged customer dataset ready for downstream analytics.
    """

    merged = cleaned_df.merge(
        explanations_df,
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    ).merge(
        clustered_df[[ID_COLUMN, "cluster_id", "segment_name"]],
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    ordered_columns = [
        ID_COLUMN,
        "churn_probability",
        "reason_1",
        "reason_2",
        "reason_3",
        "segment_name",
    ] + [column for column in cleaned_df.columns if column != ID_COLUMN]

    final_df = merged[ordered_columns].copy()
    final_df.to_csv(FINAL_OUTPUT_PATH, index=False)
    return final_df


def main() -> None:
    """Run the customer segmentation workflow end to end.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    cleaned_df = load_csv(CLEAN_DATA_PATH, "Cleaned churn dataset")
    explanations_df = load_csv(EXPLANATIONS_PATH, "Customer explanations export")

    ensure_required_columns(
        cleaned_df,
        [
            ID_COLUMN,
            "customer_satisfaction_score",
            "engagement_rate",
            "subscription_length_months",
            "monthly_income_usd",
            "payment_history_delayed",
        ],
        "Cleaned churn dataset",
    )
    ensure_required_columns(
        explanations_df,
        [ID_COLUMN, "churn_probability", "reason_1", "reason_2", "reason_3"],
        "Customer explanations export",
    )

    segmentation_frame = build_segmentation_frame(cleaned_df)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(segmentation_frame)

    metrics_df = evaluate_kmeans_range(scaled_features, K_RANGE)
    save_k_selection_charts(metrics_df)

    chosen_k, reason = choose_best_k(metrics_df)
    print("\nK-MEANS SELECTION METRICS")
    print(metrics_df.round({"inertia": 3, "silhouette_score": 4}).to_string(index=False))
    print(f"\nChosen k = {chosen_k}")
    print(reason)

    kmeans = KMeans(n_clusters=chosen_k, random_state=RANDOM_STATE, n_init="auto")
    cluster_labels = kmeans.fit_predict(scaled_features)

    clustered_df = cleaned_df.copy()
    clustered_df["cluster_id"] = cluster_labels
    clustered_df = clustered_df.merge(
        explanations_df,
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    global_reference = build_global_reference(cleaned_df, explanations_df)
    profiles_df = profile_clusters(clustered_df, global_reference)
    print_cluster_profiles(profiles_df)

    cluster_name_map = profiles_df.set_index("cluster_id")["segment_name"].to_dict()
    clustered_df["segment_name"] = clustered_df["cluster_id"].map(cluster_name_map)

    save_cluster_scatter_plot(clustered_df, scaler, kmeans, profiles_df)
    final_df = build_final_dataset(cleaned_df, explanations_df, clustered_df)

    print(f"\nSaved cluster profiles to: {SEGMENT_PROFILE_PATH}")
    print(f"Saved final customer dataset to: {FINAL_OUTPUT_PATH}")
    print(f"Saved segmentation charts to: {SEGMENTATION_CHARTS_DIR}")
    print("\nFINAL DATASET PREVIEW")
    print(final_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
