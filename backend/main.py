"""FastAPI backend for the churn analytics portfolio project."""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.explain_model import (  # noqa: E402
    PLAN_MAP,
    build_feature_metadata,
    build_reason_sentence,
    decode_one_hot_group,
    reconstruct_raw_features,
)


FINAL_DATA_PATH = PROJECT_ROOT / "outputs" / "final_customer_dataset.csv"
CLUSTER_PROFILE_PATH = PROJECT_ROOT / "outputs" / "cluster_profiles.csv"
MODEL_BUNDLE_PATH = PROJECT_ROOT / "models" / "churn_model.pkl"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DEFAULT_PORT = int(os.getenv("PORT", "8000"))

DEFAULT_PAGE_SIZE = 25
MAX_PAGE_SIZE = 100
TARGET_COLUMN = "churn_status"
ID_COLUMN = "customer_id"

DEVICE_PREFIX = "device_used_most_often_"
GENRE_PREFIX = "genre_preference_"
REGION_PREFIX = "region_"

PLAN_ORDER = ["Basic", "Standard", "Premium"]
DEVICE_ORDER = ["Desktop", "Laptop", "Mobile", "Smart TV", "Tablet"]

PAYMENT_ALIASES = {
    "delayed": "Delayed",
    "on-time": "On-Time",
    "on time": "On-Time",
    "on_time": "On-Time",
}

PLAN_ALIASES = {
    "basic": "Basic",
    "standard": "Standard",
    "premium": "Premium",
}

DEVICE_ALIASES = {
    "desktop": "Desktop",
    "laptop": "Laptop",
    "mobile": "Mobile",
    "smart tv": "Smart TV",
    "smarttv": "Smart TV",
    "tablet": "Tablet",
}

GENRE_ALIASES = {
    "action": "Action",
    "comedy": "Comedy",
    "documentary": "Documentary",
    "drama": "Drama",
    "romance": "Romance",
    "sci fi": "Sci-Fi",
    "sci-fi": "Sci-Fi",
    "thriller": "Thriller",
}

REGION_ALIASES = {
    "africa": "Africa",
    "asia": "Asia",
    "europe": "Europe",
    "north america": "North America",
    "south america": "South America",
}


@dataclass
class AppState:
    """Container for the data and model artifacts loaded at startup."""

    final_df: pd.DataFrame
    display_df: pd.DataFrame
    cluster_profiles: pd.DataFrame
    model_bundle: Dict[str, Any]
    model: Any
    preprocessor: Any
    classifier: Any
    raw_features: pd.DataFrame
    customer_ids: pd.Series
    customer_id_to_index: Dict[str, int]
    feature_names: List[str]
    group_to_indices: Dict[str, List[int]]
    explainer: shap.LinearExplainer


APP_STATE: AppState | None = None


class PredictRequest(BaseModel):
    """Validated request body for a churn prediction."""

    model_config = ConfigDict(extra="forbid")

    subscription_length_months: int = Field(..., ge=1, le=60)
    customer_satisfaction_score: int = Field(..., ge=1, le=10)
    daily_watch_time_hours: float = Field(..., ge=0)
    engagement_rate: int = Field(..., ge=1, le=10)
    device_used_most_often: str = Field(..., description="Desktop, Laptop, Mobile, Smart TV, or Tablet")
    genre_preference: str = Field(..., description="Action, Comedy, Documentary, Drama, Romance, Sci-Fi, or Thriller")
    region: str = Field(..., description="Africa, Asia, Europe, North America, or South America")
    payment_history: str = Field(..., description="On-Time or Delayed")
    subscription_plan: str = Field(..., description="Basic, Standard, or Premium")
    support_queries_logged: int = Field(..., ge=0)
    age: int = Field(..., ge=13)
    monthly_income_usd: float = Field(..., ge=0)
    promotional_offers_used: int = Field(..., ge=0)
    number_of_profiles_created: int = Field(..., ge=1)

    @field_validator("subscription_plan", mode="before")
    @classmethod
    def normalize_plan(cls, value: Any) -> str:
        """Normalize subscription plan values to the fitted training labels."""

        return normalize_choice(value, PLAN_ALIASES, "subscription_plan")

    @field_validator("payment_history", mode="before")
    @classmethod
    def normalize_payment_history(cls, value: Any) -> str:
        """Normalize payment history values to the fitted training labels."""

        return normalize_choice(value, PAYMENT_ALIASES, "payment_history")

    @field_validator("device_used_most_often", mode="before")
    @classmethod
    def normalize_device(cls, value: Any) -> str:
        """Normalize device values to the fitted training labels."""

        return normalize_choice(value, DEVICE_ALIASES, "device_used_most_often")

    @field_validator("genre_preference", mode="before")
    @classmethod
    def normalize_genre(cls, value: Any) -> str:
        """Normalize genre values to the fitted training labels."""

        return normalize_choice(value, GENRE_ALIASES, "genre_preference")

    @field_validator("region", mode="before")
    @classmethod
    def normalize_region(cls, value: Any) -> str:
        """Normalize region values to the fitted training labels."""

        return normalize_choice(value, REGION_ALIASES, "region")


class PredictionResponse(BaseModel):
    """Validated response body for churn predictions."""

    churn_probability: float
    reason_1: str
    reason_2: str
    reason_3: str


def load_csv(path: Path, description: str) -> pd.DataFrame:
    """Load a CSV file and raise a useful error if it is missing."""

    if not path.exists():
        raise FileNotFoundError(f"{description} not found at {path}.")
    return pd.read_csv(path)


def load_model_bundle(path: Path) -> Dict[str, Any]:
    """Load the serialized model bundle saved by the training step."""

    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found at {path}.")
    return joblib.load(path)


def normalize_choice(value: Any, aliases: Dict[str, str], field_name: str) -> str:
    """Normalize free-text categorical input to a canonical label."""

    if value is None:
        raise ValueError(f"{field_name} is required.")
    text = str(value).strip()
    normalized = aliases.get(text.lower())
    if normalized is not None:
        return normalized
    # Fall back to a friendly title-case form if the value already matches the
    # training vocabulary but uses different capitalization.
    title_case = text.title()
    if title_case in aliases.values():
        return title_case
    raise ValueError(
        f"Invalid {field_name} value '{value}'. "
        f"Expected one of: {sorted(set(aliases.values()))}."
    )


def build_display_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add human-readable labels to the final customer dataframe."""

    display_df = df.copy()
    display_df["subscription_plan_label"] = display_df["subscription_plan"].map(PLAN_MAP)
    display_df["region_label"] = decode_one_hot_group(display_df, REGION_PREFIX)
    display_df["device_label"] = decode_one_hot_group(display_df, DEVICE_PREFIX)
    display_df["genre_label"] = decode_one_hot_group(display_df, GENRE_PREFIX)
    display_df["payment_history_label"] = np.where(
        display_df["payment_history_delayed"].astype(int) == 1,
        "Delayed",
        "On-Time",
    )
    return display_df


def aggregate_group_contributions(shap_row: np.ndarray, group_to_indices: Dict[str, List[int]]) -> Dict[str, float]:
    """Aggregate transformed SHAP values back to the original feature groups."""

    return {
        group_name: float(np.asarray(shap_row)[indices].sum())
        for group_name, indices in group_to_indices.items()
    }


def build_shap_reason_payloads(raw_row: pd.Series, shap_row: np.ndarray, group_to_indices: Dict[str, List[int]]) -> List[Dict[str, Any]]:
    """Build structured SHAP reason payloads for a customer."""

    grouped = aggregate_group_contributions(shap_row, group_to_indices)
    top_groups = sorted(grouped.items(), key=lambda item: abs(item[1]), reverse=True)[:3]

    payloads: List[Dict[str, Any]] = []
    for index, (group_name, contribution) in enumerate(top_groups, start=1):
        payloads.append(
            {
                "label": f"Reason {index}",
                "field": group_name,
                "text": build_reason_sentence(raw_row, group_name, contribution),
                "shap_value": float(contribution),
                "magnitude": float(abs(contribution)),
            }
        )
    return payloads


def extract_shap_array(shap_values: Any) -> np.ndarray:
    """Convert SHAP output into a predictable 2D numpy array."""

    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    array = np.asarray(shap_values)
    return np.atleast_2d(array)


def create_summary_breakdown(df: pd.DataFrame, group_column: str, order: List[str] | None = None) -> List[Dict[str, Any]]:
    """Compute customer count and churn rate by a categorical column."""

    grouped = (
        df.groupby(group_column, dropna=False)
        .agg(
            customer_count=(TARGET_COLUMN, "size"),
            churn_rate_pct=(TARGET_COLUMN, lambda series: float(series.mean() * 100.0)),
        )
        .reset_index()
    )
    grouped.rename(columns={group_column: "category"}, inplace=True)

    if order is not None:
        grouped["category"] = pd.Categorical(grouped["category"], categories=order, ordered=True)
        grouped = grouped.sort_values("category").reset_index(drop=True)
        grouped["category"] = grouped["category"].astype(str)
    else:
        grouped = grouped.sort_values("churn_rate_pct", ascending=False).reset_index(drop=True)

    grouped["customer_count"] = grouped["customer_count"].astype(int)
    grouped["churn_rate_pct"] = grouped["churn_rate_pct"].round(2)
    return grouped.to_dict(orient="records")


def serialize_segment_profiles(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Format cluster profile rows for the API response."""

    profiles = df.copy()
    profiles["pct_of_total"] = (profiles["pct_of_total"] * 100.0).round(1)
    profiles["avg_churn_probability"] = profiles["avg_churn_probability"].round(4)
    profiles["avg_satisfaction_score"] = profiles["avg_satisfaction_score"].round(2)
    profiles["avg_engagement_rate"] = profiles["avg_engagement_rate"].round(2)
    profiles["pct_delayed_payments"] = (profiles["pct_delayed_payments"] * 100.0).round(1)
    profiles["avg_subscription_length_months"] = profiles["avg_subscription_length_months"].round(2)
    profiles["avg_monthly_income_usd"] = profiles["avg_monthly_income_usd"].round(0).astype(int)
    return profiles.to_dict(orient="records")


def build_prediction_frame(payload: PredictRequest) -> pd.DataFrame:
    """Convert a validated request model into a single-row dataframe."""

    return pd.DataFrame([payload.model_dump()])


def to_python_value(value: Any) -> Any:
    """Convert numpy/pandas scalar values to plain Python types."""

    if isinstance(value, np.generic):
        return value.item()
    return value


def build_raw_feature_values(raw_row: pd.Series) -> Dict[str, Any]:
    """Build the raw feature payload used in the customer detail modal."""

    return {
        "subscription_length_months": int(raw_row["subscription_length_months"]),
        "customer_satisfaction_score": int(raw_row["customer_satisfaction_score"]),
        "daily_watch_time_hours": float(raw_row["daily_watch_time_hours"]),
        "engagement_rate": int(raw_row["engagement_rate"]),
        "device_used_most_often": str(raw_row["device_used_most_often"]),
        "genre_preference": str(raw_row["genre_preference"]),
        "region": str(raw_row["region"]),
        "payment_history": str(raw_row["payment_history"]),
        "subscription_plan": str(raw_row["subscription_plan"]),
        "support_queries_logged": int(raw_row["support_queries_logged"]),
        "age": int(raw_row["age"]),
        "monthly_income_usd": float(raw_row["monthly_income_usd"]),
        "promotional_offers_used": int(raw_row["promotional_offers_used"]),
        "number_of_profiles_created": int(raw_row["number_of_profiles_created"]),
    }


def explain_prediction(raw_row: pd.Series, transformed_row: np.ndarray, state: AppState) -> tuple[PredictionResponse, List[Dict[str, Any]]]:
    """Generate churn probability and plain-language reasons for one customer."""

    shap_values = extract_shap_array(state.explainer.shap_values(transformed_row))
    reason_payloads = build_shap_reason_payloads(raw_row, shap_values[0], state.group_to_indices)
    reasons = [item["text"] for item in reason_payloads]
    probability = float(state.model.predict_proba(pd.DataFrame([raw_row.to_dict()]))[0, 1])

    return PredictionResponse(
        churn_probability=probability,
        reason_1=reasons[0] if len(reasons) > 0 else "",
        reason_2=reasons[1] if len(reasons) > 1 else "",
        reason_3=reasons[2] if len(reasons) > 2 else "",
    ), reason_payloads


def get_state() -> AppState:
    """Return the loaded application state or raise if startup failed."""

    if APP_STATE is None:
        raise RuntimeError("Application state is not initialized.")
    return APP_STATE


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load datasets, model artifacts, and SHAP explainer once at startup."""

    global APP_STATE

    final_df = load_csv(FINAL_DATA_PATH, "Final customer dataset")
    cluster_profiles = load_csv(CLUSTER_PROFILE_PATH, "Cluster profiles")
    bundle = load_model_bundle(MODEL_BUNDLE_PATH)

    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    classifier = model.named_steps["classifier"]

    if bundle.get("best_model_name") != "Logistic Regression":
        print(
            "Warning: the saved bundle is not marked as Logistic Regression. "
            "The API expects the linear model for SHAP explanations."
        )

    raw_features, _, customer_ids = reconstruct_raw_features(final_df)
    feature_names = list(preprocessor.get_feature_names_out())
    _, group_to_indices, _ = build_feature_metadata(feature_names)
    customer_id_to_index = {str(customer_id): idx for idx, customer_id in enumerate(customer_ids.tolist())}

    # The full transformed customer base is used as the background dataset so
    # the SHAP explanations reflect the same customer distribution the API serves.
    transformed_background = preprocessor.transform(raw_features)
    explainer = shap.LinearExplainer(classifier, transformed_background)

    APP_STATE = AppState(
        final_df=final_df,
        display_df=build_display_frame(final_df),
        cluster_profiles=cluster_profiles,
        model_bundle=bundle,
        model=model,
        preprocessor=preprocessor,
        classifier=classifier,
        raw_features=raw_features,
        customer_ids=customer_ids,
        customer_id_to_index=customer_id_to_index,
        feature_names=feature_names,
        group_to_indices=group_to_indices,
        explainer=explainer,
    )

    yield
    APP_STATE = None


app = FastAPI(
    title="AI-Powered Subscription Churn Prediction & Retention Analytics",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check() -> Dict[str, str]:
    """Simple health check endpoint for uptime checks."""

    return {"status": "ok", "message": "Churn analytics API is running."}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the dashboard landing page."""

    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/summary")
def get_summary() -> Dict[str, Any]:
    """Return aggregate KPI values and churn breakdowns."""

    state = get_state()
    df = state.display_df

    at_risk_mask = df["churn_probability"] > 0.5
    at_risk_revenue = float(df.loc[at_risk_mask, "monthly_income_usd"].sum())

    return {
        "total_customers": int(len(df)),
        "overall_churn_rate_pct": round(float(df[TARGET_COLUMN].mean() * 100.0), 2),
        "at_risk_customer_count": int(at_risk_mask.sum()),
        "estimated_at_risk_revenue": round(at_risk_revenue, 2),
        "breakdowns": {
            "segment_name": create_summary_breakdown(
                df,
                "segment_name",
                order=state.cluster_profiles.sort_values("cluster_id")["segment_name"].tolist(),
            ),
            "subscription_plan": create_summary_breakdown(df, "subscription_plan_label", order=PLAN_ORDER),
            "region": create_summary_breakdown(df, "region_label"),
            "device_used_most_often": create_summary_breakdown(df, "device_label", order=DEVICE_ORDER),
        },
    }


@app.get("/customers")
def list_customers(
    page: int = Query(1, ge=1),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    sort_by: str = Query("churn_probability"),
    order: Literal["asc", "desc"] = "desc",
) -> Dict[str, Any]:
    """Return a paginated customer list with optional churn sorting."""

    state = get_state()
    df = state.display_df.copy()

    if sort_by != "churn_probability":
        raise HTTPException(status_code=400, detail="Only sort_by=churn_probability is supported.")

    df = df.sort_values("churn_probability", ascending=(order == "asc"), kind="mergesort")

    total = len(df)
    start = (page - 1) * page_size
    end = start + page_size
    page_df = df.iloc[start:end]

    items = (
        page_df[
            [
                ID_COLUMN,
                "churn_probability",
                "segment_name",
                "reason_1",
                "reason_2",
                "reason_3",
                "subscription_plan_label",
                "region_label",
            ]
        ]
        .rename(columns={"subscription_plan_label": "subscription_plan", "region_label": "region"})
        .to_dict(orient="records")
    )

    return {
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": int(np.ceil(total / page_size)),
        "items": items,
    }


@app.get("/customers/{customer_id}")
def get_customer(customer_id: str) -> Dict[str, Any]:
    """Return the full record and explanations for one customer."""

    state = get_state()
    customer_index = state.customer_id_to_index.get(customer_id)
    if customer_index is None:
        raise HTTPException(status_code=404, detail=f"Customer ID '{customer_id}' was not found.")

    raw_row = state.raw_features.iloc[customer_index]
    display_row = state.display_df.iloc[customer_index]
    transformed_row = state.preprocessor.transform(state.raw_features.iloc[[customer_index]])
    shap_values = extract_shap_array(state.explainer.shap_values(transformed_row))
    shap_reasons = build_shap_reason_payloads(raw_row, shap_values[0], state.group_to_indices)

    return {
        "customer_id": str(display_row[ID_COLUMN]),
        "churn_probability": float(display_row["churn_probability"]),
        "segment_name": str(display_row["segment_name"]),
        "reason_1": shap_reasons[0]["text"] if len(shap_reasons) > 0 else "",
        "reason_2": shap_reasons[1]["text"] if len(shap_reasons) > 1 else "",
        "reason_3": shap_reasons[2]["text"] if len(shap_reasons) > 2 else "",
        "reason_1_strength": shap_reasons[0]["magnitude"] if len(shap_reasons) > 0 else 0.0,
        "reason_2_strength": shap_reasons[1]["magnitude"] if len(shap_reasons) > 1 else 0.0,
        "reason_3_strength": shap_reasons[2]["magnitude"] if len(shap_reasons) > 2 else 0.0,
        "shap_reasons": shap_reasons,
        "raw_features": build_raw_feature_values(raw_row),
        "features": {
            column: to_python_value(display_row[column])
            for column in state.final_df.columns
            if column not in {ID_COLUMN, "churn_probability", "reason_1", "reason_2", "reason_3", "segment_name"}
        },
        "decoded": {
            "subscription_plan": str(display_row["subscription_plan_label"]),
            "region": str(display_row["region_label"]),
            "device_used_most_often": str(display_row["device_label"]),
            "genre_preference": str(display_row["genre_label"]),
            "payment_history": str(display_row["payment_history_label"]),
        },
    }


@app.get("/segments")
def get_segments() -> Dict[str, Any]:
    """Return the stored cluster profile summary."""

    state = get_state()
    return {"segments": serialize_segment_profiles(state.cluster_profiles)}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictRequest) -> PredictionResponse:
    """Predict churn for a new customer and explain the result with SHAP."""

    state = get_state()
    raw_row = pd.Series(request.model_dump())
    raw_frame = build_prediction_frame(request)
    transformed_row = state.preprocessor.transform(raw_frame)
    prediction, _reason_payloads = explain_prediction(raw_row, transformed_row, state)
    return prediction


app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host="0.0.0.0", port=DEFAULT_PORT)
