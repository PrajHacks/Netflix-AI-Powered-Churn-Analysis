# Churn Project Scripts

This folder contains the standalone scripts used for the first two phases of the
project so the work can be reproduced without notebook state.

## Script Order

1. `python src/clean_data.py`
2. `python src/eda.py`

Run them in that order because `eda.py` depends on the cleaned CSV created by
`clean_data.py`.

## `clean_data.py`

### What it does

- loads the raw churn dataset
- standardizes column names to snake_case
- handles missing values with feature-specific rules
- ordinal-encodes `subscription_plan`
- one-hot encodes the nominal categorical columns
- binary-encodes the churn target
- writes the cleaned file to `outputs/cleaned_churn_data.csv`

### Input

- Preferred: `data/churn_data.csv`
- Workspace fallback: `dataset/netflix_large_user_data.xlsx`

### Output

- `outputs/cleaned_churn_data.csv`

### Cleaning assumptions

- No missing values were present in the supplied dataset, so the imputation
  logic did not change the current data.
- If missing values appear later, the script uses:
  - median imputation for numeric columns
  - mode imputation for `subscription_plan`
  - a literal `Missing` category for nominal categoricals before one-hot
    encoding
  - row drops for missing `customer_id` or `churn_status`
- `subscription_plan` is treated as an ordered field:
  - Basic = 1
  - Standard = 2
  - Premium = 3
- `churn_status` is encoded as:
  - No = 0
  - Yes = 1

## `eda.py`

### What it does

- loads `outputs/cleaned_churn_data.csv`
- reconstructs human-readable categorical labels from the encoded columns
- creates churn breakdown charts by plan, region, device, and payment history
- creates box plots for satisfaction, engagement, watch time, and support
  queries
- creates a numeric correlation heatmap
- creates a top-correlations chart
- creates a compounding-effect chart for satisfaction and payment history
- saves all charts into `eda_charts/`
- prints a plain-language summary to the console

### Input

- `outputs/cleaned_churn_data.csv`

### Output

- PNG charts in `eda_charts/`
- console summary of the main findings

## Notes

- The scripts use path constants at the top so the folder locations are easy to
  change later.
- Both scripts are safe to run directly with `python ...` thanks to their
  `if __name__ == "__main__": main()` blocks.
- The EDA script saves both the detailed charts requested in the analysis
  phase and two small overview figures that are useful for quick presentation
  review.
