"""Generate exploratory data analysis charts and a plain-language summary.

Run this script directly from the command line:

    python eda.py

What it does:
- loads ``outputs/cleaned_churn_data.csv``
- reconstructs the human-readable categorical labels from the encoded data
- saves the churn breakdown charts, box plots, heatmap, and compounding-effect
  view into ``eda_charts/``
- prints a short stakeholder-friendly summary to the console
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DATA_PATH = PROJECT_ROOT / "outputs" / "cleaned_churn_data.csv"
EDA_CHARTS_DIR = PROJECT_ROOT / "eda_charts"

PLAN_LABELS = {
    1: "Basic",
    2: "Standard",
    3: "Premium",
}

PAYMENT_ORDER = ["On-Time", "Delayed"]
DEVICE_ORDER = ["Laptop", "Desktop", "Mobile", "Tablet", "Smart TV"]
REGION_ORDER = ["Africa", "Asia", "Europe", "North America", "South America"]
PLAN_ORDER = ["Basic", "Standard", "Premium"]

CORRELATION_LABEL_MAP = {
    "age": "Age",
    "genre_preference_comedy": "Genre: Comedy",
    "device_used_most_often_laptop": "Device: Laptop",
    "support_queries_logged": "Support Queries",
    "device_used_most_often_smart_tv": "Device: Smart TV",
    "region_south_america": "Region: South America",
    "genre_preference_thriller": "Genre: Thriller",
    "monthly_income_usd": "Monthly Income",
    "device_used_most_often_tablet": "Device: Tablet",
    "customer_satisfaction_score": "Satisfaction Score",
    "device_used_most_often_desktop": "Device: Desktop",
    "genre_preference_romance": "Genre: Romance",
}


def load_cleaned_data(clean_path: Path) -> pd.DataFrame:
    """Load the cleaned churn dataset from CSV.

    Parameters
    ----------
    clean_path:
        Path to the cleaned CSV created by ``clean_data.py``.

    Returns
    -------
    pandas.DataFrame
        Cleaned and encoded churn data.
    """

    if not clean_path.exists():
        raise FileNotFoundError(
            f"Cleaned data not found at {clean_path}. "
            "Run clean_data.py first."
        )
    return pd.read_csv(clean_path)


def ensure_output_directory(output_dir: Path) -> None:
    """Create the chart output directory if it does not already exist.

    Parameters
    ----------
    output_dir:
        Destination directory for saved figures.

    Returns
    -------
    None
    """

    output_dir.mkdir(parents=True, exist_ok=True)


def format_category_label(value: str) -> str:
    """Convert decoded category strings into polished display labels.

    Parameters
    ----------
    value:
        A raw decoded label such as ``smart tv`` or ``on time``.

    Returns
    -------
    str
        A business-friendly label such as ``Smart TV`` or ``On-Time``.
    """

    display = value.replace("_", " ").title()
    display = display.replace("Sci Fi", "Sci-Fi")
    display = display.replace("On Time", "On-Time")
    display = display.replace("Smart Tv", "Smart TV")
    return display


def decode_one_hot_group(df: pd.DataFrame, prefix: str) -> pd.Series:
    """Recover the original categorical values from a one-hot encoded group.

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
        Human-readable categorical labels for each row.
    """

    columns = [column for column in df.columns if column.startswith(prefix)]
    decoded = df[columns].idxmax(axis=1).str.replace(prefix, "", regex=False)
    return decoded.map(format_category_label)


def reconstruct_human_readable_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add the readable categorical labels needed for EDA charts.

    Parameters
    ----------
    df:
        Encoded churn dataframe from ``clean_data.py``.

    Returns
    -------
    pandas.DataFrame
        A copy of the dataframe with helper columns for plotting.
    """

    df = df.copy()
    df["subscription_plan_label"] = df["subscription_plan"].map(PLAN_LABELS)
    df["device_used_most_often"] = decode_one_hot_group(df, "device_used_most_often_")
    df["genre_preference"] = decode_one_hot_group(df, "genre_preference_")
    df["region"] = decode_one_hot_group(df, "region_")
    df["payment_history"] = decode_one_hot_group(df, "payment_history_")
    df["churn_label"] = df["churn_status"].map({0: "No", 1: "Yes"})
    return df


def save_bar_chart(
    df: pd.DataFrame,
    group_column: str,
    order: list[str],
    title: str,
    x_label: str,
    file_name: str,
    palette: list[str] | None = None,
    rotation: int = 0,
) -> None:
    """Create and save a churn-rate bar chart for a categorical segment.

    Parameters
    ----------
    df:
        Dataframe containing the decoded categories and churn label.
    group_column:
        Column to segment by.
    order:
        Explicit ordering of the category labels.
    title:
        Chart title.
    x_label:
        X-axis label.
    file_name:
        Output filename inside ``eda_charts/``.
    palette:
        Optional list of bar colors.
    rotation:
        X-axis tick rotation in degrees.

    Returns
    -------
    None
    """

    churn_rates = df.groupby(group_column)["churn_status"].mean().reindex(order) * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = palette or sns.color_palette("viridis", len(churn_rates))
    bars = ax.bar(churn_rates.index, churn_rates.values, color=colors)
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_ylim(0, max(churn_rates.values) * 1.25)
    ax.tick_params(axis="x", rotation=rotation)
    for bar, value in zip(bars, churn_rates.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(EDA_CHARTS_DIR / file_name, bbox_inches="tight")
    plt.close(fig)


def save_behavior_boxplot(
    df: pd.DataFrame,
    column: str,
    title: str,
    y_label: str,
    file_name: str,
) -> None:
    """Save a box plot comparing churned and non-churned customers.

    Parameters
    ----------
    df:
        Dataframe containing the decoded categories and churn label.
    column:
        Numeric feature to compare.
    title:
        Plot title.
    y_label:
        Y-axis label.
    file_name:
        Output filename inside ``eda_charts/``.

    Returns
    -------
    None
    """

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sns.boxplot(
        data=df,
        x="churn_label",
        y=column,
        hue="churn_label",
        dodge=False,
        palette={"No": "#2ca02c", "Yes": "#d62728"},
        ax=ax,
    )
    if ax.legend_ is not None:
        ax.legend_.remove()
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Churn Status")
    ax.set_ylabel(y_label)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(EDA_CHARTS_DIR / file_name, bbox_inches="tight")
    plt.close(fig)


def save_correlation_heatmap(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    """Create and save the correlation heatmap for all numeric columns.

    Parameters
    ----------
    df:
        Encoded churn dataframe.
    file_name:
        Output filename inside ``eda_charts/``.

    Returns
    -------
    pandas.DataFrame
        The correlation matrix, which can be reused for summary analysis.
    """

    numeric_df = df.select_dtypes(include="number").copy()
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(18, 15))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=False,
        cbar_kws={"shrink": 0.7},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap of All Numeric Features", fontweight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.tight_layout()
    fig.savefig(EDA_CHARTS_DIR / file_name, bbox_inches="tight")
    plt.close(fig)

    return corr


def save_top_correlations_plot(
    corr: pd.DataFrame,
    file_name: str,
    top_n: int = 12,
) -> pd.Series:
    """Save a horizontal bar chart of the strongest churn correlations.

    Parameters
    ----------
    corr:
        Correlation matrix from ``save_correlation_heatmap``.
    file_name:
        Output filename inside ``eda_charts/``.
    top_n:
        Number of features to display.

    Returns
    -------
    pandas.Series
        Correlations with churn sorted by absolute magnitude.
    """

    churn_corr = corr["churn_status"].drop("churn_status").sort_values(
        key=lambda series: series.abs(),
        ascending=False,
    )

    top_corr = churn_corr.head(top_n).rename(
        index=lambda value: CORRELATION_LABEL_MAP.get(
            value,
            value.replace("_", " ").title(),
        )
    )

    fig, ax = plt.subplots(figsize=(11, 7.2))
    colors = ["#d62728" if value > 0 else "#1f77b4" for value in top_corr]
    ax.barh(top_corr.index[::-1], top_corr.values[::-1], color=colors[::-1])
    ax.set_title("Top Correlations with Churn Status", fontweight="bold")
    ax.set_xlabel("Correlation")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlim(-0.065, 0.055)
    ax.tick_params(axis="y", labelsize=11)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(EDA_CHARTS_DIR / file_name, bbox_inches="tight")
    plt.close(fig)

    return churn_corr


def save_compounding_effect_plot(df: pd.DataFrame, file_name: str) -> pd.DataFrame:
    """Save the satisfaction-and-payment compounding-effect chart.

    Parameters
    ----------
    df:
        Dataframe containing the decoded categories and churn label.
    file_name:
        Output filename inside ``eda_charts/``.

    Returns
    -------
    pandas.DataFrame
        Pivot table of churn rate percentages by satisfaction bucket and
        payment history.
    """

    # A threshold of 4 keeps the "low satisfaction" bucket easy to explain to
    # non-technical stakeholders while still being broad enough to be useful.
    df = df.copy()
    df["satisfaction_bucket"] = np.where(
        df["customer_satisfaction_score"] <= 4,
        "Low Satisfaction (1-4)",
        "Higher Satisfaction (5-10)",
    )

    combo = (
        df.groupby(["satisfaction_bucket", "payment_history"])["churn_status"]
        .agg(["mean", "count"])
        .reset_index()
    )
    combo["churn_rate_pct"] = combo["mean"] * 100

    pivot = combo.pivot(
        index="satisfaction_bucket",
        columns="payment_history",
        values="churn_rate_pct",
    ).reindex(["Low Satisfaction (1-4)", "Higher Satisfaction (5-10)"])

    fig, ax = plt.subplots(figsize=(9, 6))
    x_positions = np.arange(len(PAYMENT_ORDER))
    satisfaction_levels = ["Low Satisfaction (1-4)", "Higher Satisfaction (5-10)"]
    colors = ["#d62728", "#2ca02c"]

    for offset, satisfaction_level in enumerate(satisfaction_levels):
        subset = combo[combo["satisfaction_bucket"] == satisfaction_level].copy()
        subset["payment_history"] = pd.Categorical(
            subset["payment_history"],
            categories=PAYMENT_ORDER,
            ordered=True,
        )
        subset = subset.sort_values("payment_history")
        bar_positions = x_positions + (offset - 0.5) * 0.35
        ax.bar(
            bar_positions,
            subset["churn_rate_pct"].values,
            width=0.35,
            label=satisfaction_level,
            color=colors[offset],
            alpha=0.85,
        )
        for x_pos, value in zip(bar_positions, subset["churn_rate_pct"].values):
            ax.text(
                x_pos,
                value + 0.8,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(PAYMENT_ORDER)
    ax.set_title("Compounding Effect: Satisfaction and Payment History", fontweight="bold")
    ax.set_xlabel("Payment History")
    ax.set_ylabel("Churn Rate (%)")
    ax.legend(title="Satisfaction Bucket")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(EDA_CHARTS_DIR / file_name, bbox_inches="tight")
    plt.close(fig)

    return pivot


def save_overview_breakdowns(df: pd.DataFrame) -> None:
    """Save a 2x2 overview figure for the churn-rate breakdowns.

    Parameters
    ----------
    df:
        Dataframe containing the decoded categories and churn label.

    Returns
    -------
    None
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Churn Rate Breakdowns", fontsize=20, fontweight="bold", y=1.02)

    region_order = (
        df.groupby("region")["churn_status"].mean().sort_values(ascending=False).index.tolist()
    )
    device_order = (
        df.groupby("device_used_most_often")["churn_status"].mean().sort_values(ascending=False).index.tolist()
    )

    breakdown_specs = [
        (
            axes[0, 0],
            "subscription_plan_label",
            PLAN_ORDER,
            "By Subscription Plan",
            "Subscription Plan",
            ["#4e79a7", "#f28e2b", "#59a14f"],
            0,
        ),
        (
            axes[0, 1],
            "region",
            region_order,
            "By Region",
            "Region",
            sns.color_palette("crest", len(region_order)),
            20,
        ),
        (
            axes[1, 0],
            "device_used_most_often",
            device_order,
            "By Device Used Most Often",
            "Device",
            sns.color_palette("mako", len(device_order)),
            15,
        ),
        (
            axes[1, 1],
            "payment_history",
            PAYMENT_ORDER,
            "By Payment History",
            "Payment History",
            ["#2ca02c", "#d62728"],
            0,
        ),
    ]

    for ax, group_column, order, title, x_label, colors, rotation in breakdown_specs:
        churn_rates = df.groupby(group_column)["churn_status"].mean().reindex(order) * 100
        bars = ax.bar(churn_rates.index, churn_rates.values, color=colors)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Churn Rate (%)")
        ax.set_ylim(0, max(churn_rates.values) * 1.25)
        ax.tick_params(axis="x", rotation=rotation)
        for bar, value in zip(bars, churn_rates.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        sns.despine(ax=ax)

    fig.tight_layout()
    fig.savefig(EDA_CHARTS_DIR / "12_churn_breakdowns_overview.png", bbox_inches="tight")
    plt.close(fig)


def save_behavior_overview(df: pd.DataFrame) -> None:
    """Save a 2x2 overview figure for the behavioral box plots.

    Parameters
    ----------
    df:
        Dataframe containing the decoded categories and churn label.

    Returns
    -------
    None
    """

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Behavioral Relationships with Churn", fontsize=20, fontweight="bold", y=1.02)

    specs = [
        ("customer_satisfaction_score", "Customer Satisfaction Score"),
        ("engagement_rate", "Engagement Rate"),
        ("daily_watch_time_hours", "Daily Watch Time (Hours)"),
        ("support_queries_logged", "Support Queries Logged"),
    ]

    for ax, (column, title) in zip(axes.flat, specs):
        sns.boxplot(
            data=df,
            x="churn_label",
            y=column,
            hue="churn_label",
            dodge=False,
            palette={"No": "#2ca02c", "Yes": "#d62728"},
            ax=ax,
        )
        if ax.legend_ is not None:
            ax.legend_.remove()
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Churn Status")
        ax.set_ylabel(title)
        sns.despine(ax=ax)

    fig.tight_layout()
    fig.savefig(EDA_CHARTS_DIR / "13_behavior_overview.png", bbox_inches="tight")
    plt.close(fig)


def print_summary_findings(
    df: pd.DataFrame,
    churn_corr: pd.Series,
    pivot: pd.DataFrame,
) -> None:
    """Print the stakeholder-facing summary of the EDA findings.

    Parameters
    ----------
    df:
        Decoded dataframe used for analysis.
    churn_corr:
        Series of correlations with churn, sorted by absolute magnitude.
    pivot:
        Pivot table from the compounding-effect analysis.

    Returns
    -------
    None
    """

    overall_churn = df["churn_status"].mean() * 100
    plan_rates = (df.groupby("subscription_plan_label")["churn_status"].mean() * 100).reindex(PLAN_ORDER)
    region_rates = (df.groupby("region")["churn_status"].mean() * 100).reindex(REGION_ORDER)
    device_rates = (df.groupby("device_used_most_often")["churn_status"].mean() * 100).reindex(DEVICE_ORDER)
    payment_rates = (df.groupby("payment_history")["churn_status"].mean() * 100).reindex(PAYMENT_ORDER)

    top_corr_feature = churn_corr.index[0]
    top_corr_value = churn_corr.iloc[0]
    max_combo = pivot.max().max()
    min_combo = pivot.min().min()

    summary_lines = [
        f"- Overall churn is {overall_churn:.1f}%, so churn is more common than retention in this dataset.",
        (
            f"- Subscription plan barely changes churn: "
            f"{plan_rates.idxmax()} is {plan_rates.max():.1f}% versus "
            f"{plan_rates.idxmin()} at {plan_rates.min():.1f}%."
        ),
        (
            f"- Device usage has the clearest segment spread: "
            f"{device_rates.idxmax()} is highest at {device_rates.max():.1f}%, "
            f"while {device_rates.idxmin()} is lowest at {device_rates.min():.1f}%."
        ),
        (
            f"- Regional differences are present but modest, ranging from "
            f"{region_rates.min():.1f}% in {region_rates.idxmin()} to "
            f"{region_rates.max():.1f}% in {region_rates.idxmax()}."
        ),
        (
            f"- Payment history is almost flat on its own, with On-Time at "
            f"{payment_rates['On-Time']:.1f}% and Delayed at {payment_rates['Delayed']:.1f}%."
        ),
        (
            f"- The strongest linear correlation with churn is still weak: "
            f"{CORRELATION_LABEL_MAP.get(top_corr_feature, top_corr_feature)} at {top_corr_value:.3f}."
        ),
        (
            f"- The compounding view does not show a dramatic low-satisfaction + delayed-payment spike; "
            f"rates range only from {min_combo:.1f}% to {max_combo:.1f}% across the four combinations."
        ),
    ]

    print("\nSUMMARY FINDINGS")
    for line in summary_lines:
        print(line)


def main() -> None:
    """Run the EDA pipeline end to end.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "font.family": "DejaVu Sans",
        }
    )

    ensure_output_directory(EDA_CHARTS_DIR)
    df = load_cleaned_data(CLEAN_DATA_PATH)
    df = reconstruct_human_readable_columns(df)

    save_bar_chart(
        df,
        group_column="subscription_plan_label",
        order=PLAN_ORDER,
        title="Churn Rate by Subscription Plan",
        x_label="Subscription Plan",
        file_name="01_churn_rate_by_subscription_plan.png",
        palette=["#4e79a7", "#f28e2b", "#59a14f"],
    )
    save_bar_chart(
        df,
        group_column="region",
        order=df.groupby("region")["churn_status"].mean().sort_values(ascending=False).index.tolist(),
        title="Churn Rate by Region",
        x_label="Region",
        file_name="02_churn_rate_by_region.png",
        palette=sns.color_palette(
            "crest",
            len(df.groupby("region")["churn_status"].mean().sort_values(ascending=False)),
        ),
        rotation=20,
    )
    save_bar_chart(
        df,
        group_column="device_used_most_often",
        order=df.groupby("device_used_most_often")["churn_status"].mean().sort_values(ascending=False).index.tolist(),
        title="Churn Rate by Device Used Most Often",
        x_label="Device",
        file_name="03_churn_rate_by_device.png",
        palette=sns.color_palette(
            "mako",
            len(
                df.groupby("device_used_most_often")["churn_status"]
                .mean()
                .sort_values(ascending=False)
            ),
        ),
        rotation=15,
    )
    save_bar_chart(
        df,
        group_column="payment_history",
        order=PAYMENT_ORDER,
        title="Churn Rate by Payment History",
        x_label="Payment History",
        file_name="04_churn_rate_by_payment_history.png",
        palette=["#2ca02c", "#d62728"],
    )

    save_behavior_boxplot(
        df,
        column="customer_satisfaction_score",
        title="Customer Satisfaction Score vs Churn Status",
        y_label="Customer Satisfaction Score",
        file_name="05_satisfaction_boxplot.png",
    )
    save_behavior_boxplot(
        df,
        column="engagement_rate",
        title="Engagement Rate vs Churn Status",
        y_label="Engagement Rate",
        file_name="06_engagement_boxplot.png",
    )
    save_behavior_boxplot(
        df,
        column="daily_watch_time_hours",
        title="Daily Watch Time (Hours) vs Churn Status",
        y_label="Daily Watch Time (Hours)",
        file_name="07_watch_time_boxplot.png",
    )
    save_behavior_boxplot(
        df,
        column="support_queries_logged",
        title="Support Queries Logged vs Churn Status",
        y_label="Support Queries Logged",
        file_name="08_support_queries_boxplot.png",
    )

    corr = save_correlation_heatmap(df, "09_correlation_heatmap_numeric_features.png")
    churn_corr = save_top_correlations_plot(corr, "10_top_correlations_with_churn.png")
    pivot = save_compounding_effect_plot(df, "11_compounding_effect_satisfaction_payment.png")

    # The overview figures are handy for a README or quick stakeholder review.
    save_overview_breakdowns(df)
    save_behavior_overview(df)

    print("Saved charts to:", EDA_CHARTS_DIR)
    print_summary_findings(df, churn_corr, pivot)


if __name__ == "__main__":
    main()
