"""Clean and encode the churn dataset.

Run this script directly from the command line:

    python clean_data.py

What it does:
- loads the raw churn dataset from CSV, with an XLSX fallback for the current
  workspace
- handles missing values with sensible, feature-specific defaults
- renames columns to snake_case for easier downstream use
- ordinal-encodes the subscription plan
- one-hot encodes the nominal categorical variables
- binary-encodes the churn target
- saves the cleaned result to ``outputs/cleaned_churn_data.csv``
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "churn_data.csv"
RAW_XLSX_FALLBACK_PATH = PROJECT_ROOT / "dataset" / "netflix_large_user_data.xlsx"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "cleaned_churn_data.csv"

RAW_COLUMN_RENAME_MAP = {
    "Customer ID": "customer_id",
    "Subscription Length (Months)": "subscription_length_months",
    "Customer Satisfaction Score (1-10)": "customer_satisfaction_score",
    "Daily Watch Time (Hours)": "daily_watch_time_hours",
    "Engagement Rate (1-10)": "engagement_rate",
    "Device Used Most Often": "device_used_most_often",
    "Genre Preference": "genre_preference",
    "Region": "region",
    "Payment History (On-Time/Delayed)": "payment_history",
    "Subscription Plan": "subscription_plan",
    "Churn Status (Yes/No)": "churn_status",
    "Support Queries Logged": "support_queries_logged",
    "Age": "age",
    "Monthly Income ($)": "monthly_income_usd",
    "Promotional Offers Used": "promotional_offers_used",
    "Number of Profiles Created": "number_of_profiles_created",
}

NUMERIC_MEDIAN_COLUMNS = [
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

ORDINAL_COLUMN = "subscription_plan"
ORDINAL_COLUMN_MAP = {
    "Basic": 1,
    "Standard": 2,
    "Premium": 3,
}

TARGET_COLUMN = "churn_status"
TARGET_MAP = {
    "No": 0,
    "Yes": 1,
}

NOMINAL_COLUMNS = [
    "device_used_most_often",
    "genre_preference",
    "region",
    "payment_history",
]

EXPECTED_CATEGORY_LEVELS = {
    "device_used_most_often": ["Desktop", "Laptop", "Mobile", "Smart TV", "Tablet"],
    "genre_preference": [
        "Action",
        "Comedy",
        "Documentary",
        "Drama",
        "Romance",
        "Sci-Fi",
        "Thriller",
    ],
    "region": [
        "Africa",
        "Asia",
        "Europe",
        "North America",
        "South America",
    ],
    "payment_history": ["Delayed", "On-Time"],
}

CORE_COLUMNS = [
    "customer_id",
    "subscription_length_months",
    "customer_satisfaction_score",
    "daily_watch_time_hours",
    "engagement_rate",
    "subscription_plan",
    "support_queries_logged",
    "age",
    "monthly_income_usd",
    "promotional_offers_used",
    "number_of_profiles_created",
]

MISSING_CATEGORY_TOKEN = "Missing"


def slugify(value: str) -> str:
    """Convert a label into a lower snake_case column name.

    Parameters
    ----------
    value:
        The original label or column name.

    Returns
    -------
    str
        A normalized snake_case version of the label.
    """

    value = value.strip().lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def load_raw_dataset(raw_path: Path, fallback_xlsx_path: Path) -> pd.DataFrame:
    """Load the raw churn dataset from CSV, with an XLSX fallback if needed.

    Parameters
    ----------
    raw_path:
        Preferred CSV path for the raw dataset.
    fallback_xlsx_path:
        Backup workbook path used in the current workspace if the CSV has not
        been created yet.

    Returns
    -------
    pandas.DataFrame
        The raw churn dataset.
    """

    if raw_path.exists():
        if raw_path.suffix.lower() == ".csv":
            return pd.read_csv(raw_path)
        if raw_path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(raw_path)
        raise ValueError(f"Unsupported input format: {raw_path.suffix}")

    if fallback_xlsx_path.exists():
        print(
            f"Raw CSV not found at {raw_path}. "
            f"Falling back to {fallback_xlsx_path}."
        )
        return pd.read_excel(fallback_xlsx_path)

    raise FileNotFoundError(
        "Could not find the raw dataset. Place the CSV at "
        f"{raw_path} or the fallback workbook at {fallback_xlsx_path}."
    )


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Rename the source columns to a stable snake_case schema.

    Parameters
    ----------
    df:
        Raw dataframe using the original workbook column names.

    Returns
    -------
    pandas.DataFrame
        A copy of the dataframe with renamed columns.
    """

    return df.rename(columns=RAW_COLUMN_RENAME_MAP).copy()


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Apply feature-specific missing-value handling.

    Strategy choices:
    - identifier and target rows are dropped if missing, because they cannot
      be sensibly imputed
    - numeric features use median imputation, which is robust to skew and
      outliers
    - the ordinal subscription plan uses mode imputation, because it is an
      ordered category and the most frequent level is the safest default
    - nominal categoricals receive a literal ``Missing`` category, so any
      future gaps are preserved as a distinct signal for one-hot encoding

    Parameters
    ----------
    df:
        The renamed raw dataframe.

    Returns
    -------
    pandas.DataFrame
        A cleaned dataframe with missing values handled.
    """

    df = df.copy()

    # Rows without an identifier or target label are not useful for analysis or
    # modeling, so we remove them rather than inventing values.
    df = df.dropna(subset=["customer_id", TARGET_COLUMN])

    # Numeric fields are imputed with the median to reduce the impact of
    # unusually high or low values.
    for column in NUMERIC_MEDIAN_COLUMNS:
        if df[column].isna().any():
            df[column] = df[column].fillna(df[column].median())

    # The subscription plan has a natural order, so mode imputation is safer
    # than attempting to infer a value from neighboring categories.
    if df[ORDINAL_COLUMN].isna().any():
        mode_value = df[ORDINAL_COLUMN].mode(dropna=True).iloc[0]
        df[ORDINAL_COLUMN] = df[ORDINAL_COLUMN].fillna(mode_value)

    # Preserve missingness for nominal fields as its own category.
    for column in NOMINAL_COLUMNS:
        df[column] = df[column].fillna(MISSING_CATEGORY_TOKEN)

    return df


def build_one_hot_frame(
    series: pd.Series,
    prefix: str,
    expected_levels: list[str],
) -> pd.DataFrame:
    """Create a stable one-hot encoded dataframe for a nominal feature.

    Parameters
    ----------
    series:
        The categorical series to encode.
    prefix:
        Prefix used for the one-hot column names.
    expected_levels:
        The known category levels used to keep the output column order stable.

    Returns
    -------
    pandas.DataFrame
        One-hot encoded columns for the feature.
    """

    has_missing_category = (series == MISSING_CATEGORY_TOKEN).any()
    dummy_frame = pd.get_dummies(
        series,
        prefix=prefix,
        prefix_sep="_",
        dtype="int8",
    )
    dummy_frame.columns = [slugify(column) for column in dummy_frame.columns]

    ordered_columns = [slugify(f"{prefix}_{level}") for level in expected_levels]
    if has_missing_category:
        ordered_columns.append(slugify(f"{prefix}_{MISSING_CATEGORY_TOKEN}"))

    # If the source ever introduces a new category, keep it rather than losing
    # data. The expected columns stay first, followed by any extras.
    extra_columns = [column for column in dummy_frame.columns if column not in ordered_columns]
    ordered_columns.extend(sorted(extra_columns))

    return dummy_frame.reindex(columns=ordered_columns, fill_value=0)


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the target, ordinal field, and nominal categoricals.

    Parameters
    ----------
    df:
        Cleaned dataframe after missing-value handling.

    Returns
    -------
    pandas.DataFrame
        A fully encoded dataframe ready for EDA or modeling.
    """

    df = df.copy()

    unexpected_plan_values = sorted(set(df[ORDINAL_COLUMN].dropna().unique()) - set(ORDINAL_COLUMN_MAP))
    if unexpected_plan_values:
        raise ValueError(f"Unexpected subscription plan values: {unexpected_plan_values}")

    unexpected_target_values = sorted(set(df[TARGET_COLUMN].dropna().unique()) - set(TARGET_MAP))
    if unexpected_target_values:
        raise ValueError(f"Unexpected target values: {unexpected_target_values}")

    # Convert the ordered plan to integers so Premium > Standard > Basic.
    df[ORDINAL_COLUMN] = df[ORDINAL_COLUMN].map(ORDINAL_COLUMN_MAP)

    # Binary target encoding keeps churn as the positive class.
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(TARGET_MAP)

    dummy_frames: list[pd.DataFrame] = []
    ordered_dummy_columns: list[str] = []
    for column in NOMINAL_COLUMNS:
        dummy_frame = build_one_hot_frame(
            df[column],
            prefix=column,
            expected_levels=EXPECTED_CATEGORY_LEVELS[column],
        )
        dummy_frames.append(dummy_frame)
        ordered_dummy_columns.extend(dummy_frame.columns.tolist())

    cleaned = pd.concat([df[CORE_COLUMNS], *dummy_frames, df[[TARGET_COLUMN]]], axis=1)

    final_columns = CORE_COLUMNS + ordered_dummy_columns + [TARGET_COLUMN]
    cleaned = cleaned.reindex(columns=final_columns)

    if cleaned.isna().any().any():
        raise ValueError("Unexpected missing values remained after cleaning.")

    return cleaned


def save_cleaned_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Save the cleaned dataset to disk.

    Parameters
    ----------
    df:
        The fully cleaned and encoded dataframe.
    output_path:
        Destination CSV path.

    Returns
    -------
    None
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def main() -> None:
    """Run the full cleaning pipeline end to end.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    raw_df = load_raw_dataset(RAW_DATA_PATH, RAW_XLSX_FALLBACK_PATH)
    renamed_df = standardize_column_names(raw_df)

    print(f"Loaded raw data with shape: {raw_df.shape}")
    print(f"Missing values before cleaning: {int(raw_df.isna().sum().sum())}")

    cleaned_df = handle_missing_values(renamed_df)
    encoded_df = encode_categoricals(cleaned_df)
    save_cleaned_dataset(encoded_df, OUTPUT_PATH)

    print(f"Saved cleaned dataset to: {OUTPUT_PATH}")
    print(f"Final shape: {encoded_df.shape}")
    print(f"Missing values after cleaning: {int(encoded_df.isna().sum().sum())}")


if __name__ == "__main__":
    main()
