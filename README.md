# AI-Powered Subscription Churn Prediction & Retention Analytics

An end-to-end data science and full-stack project that predicts customer churn for a subscription service, explains *why* each customer is at risk using SHAP, segments customers into actionable retention groups, and serves it all through a live interactive dashboard.

🔗 **Live App**: (https://netflix-ai-powered-churn-analysis.onrender.com/)
📊 **Power BI Report**: <img width="1122" height="627" alt="net1" src="https://github.com/user-attachments/assets/160ddc41-ee80-4ae8-84f7-225badb7ac73" />
<img width="1121" height="626" alt="net2" src="https://github.com/user-attachments/assets/ee7737ea-4c6b-4948-9d38-06926c89e14d" />
<img width="1112" height="627" alt="net3" src="https://github.com/user-attachments/assets/4589f72f-7c2c-45e6-a164-6dd4f4637ba8" />



---

## Overview

Subscription businesses lose revenue when customers cancel — but by the time someone cancels, it's too late to act. This project builds a system that flags at-risk customers *before* they churn, explains the key drivers behind each prediction in plain language, and groups customers into segments so a retention team can act with targeted strategies instead of treating everyone the same.

This isn't just a model in a notebook — it's a full pipeline: data → trained model → explainability layer → customer segments → API → interactive dashboard with live predictions.

---

## Key Results

- **Overall churn rate**: 42.1%
- **Best model**: Logistic Regression (ROC-AUC: 0.87, Recall: 0.76) — chosen over Random Forest specifically because recall matters more here; missing an at-risk customer costs more than a false alarm
- **Top churn drivers** (via SHAP): Customer Satisfaction Score, Engagement Rate, Support Queries Logged, Subscription Length, Promotional Offers Used
- **4 customer segments identified**, ranging from "Satisfied but Passive" (23.1% of customers, 30.8% avg churn) to "High-Risk Disengaged" (24.4% of customers, 63.9% avg churn)
- Satisfaction and payment history compound: churn ranges from **27.7% to 69.2%** depending on how these two factors combine

---

## Architecture

Raw Data → Cleaning/Preprocessing → EDA → Model Training → SHAP Explainability
→ Customer Segmentation → Merged Dataset → FastAPI Backend → Web Dashboard


---

## Tech Stack

**Data Science**: Python, Pandas, scikit-learn, SHAP, Matplotlib/Seaborn
**Backend**: FastAPI, Uvicorn, Joblib
**Frontend**: HTML, CSS, JavaScript (no framework), Chart.js
**Deployment**: Render (backend), Vercel (frontend) *[or update based on your final setup]*
**Visualization**: Power BI (supplementary report)

---

## Features

- **Predictive model** trained and evaluated with cross-validation (Logistic Regression vs. Random Forest comparison)
- **SHAP-based explainability** — every customer gets a plain-language explanation, not just a probability score (e.g., *"Low satisfaction score (1/10) increased churn risk"*)
- **K-Means customer segmentation** into business-friendly, auto-labeled groups
- **REST API** exposing summary KPIs, paginated customer data, individual customer detail, segment profiles, and a live prediction endpoint
- **Interactive dashboard** with KPI cards, charts, a sortable/filterable customer table, drill-down modals, and a live "predict a new customer" form that calls the model in real time

---

## Project Structure

churn-project/
├── data/ # raw input data
├── src/
│ ├── clean_data.py # data cleaning & preprocessing
│ ├── eda.py # exploratory data analysis
│ ├── train_model.py # model training & evaluation
│ ├── explain_model.py # SHAP explainability
│ └── segment_customers.py # K-Means segmentation
├── backend/
│ └── main.py # FastAPI app
├── frontend/
│ ├── index.html
│ ├── style.css
│ └── app.js
├── models/
│ └── churn_model.pkl # trained model bundle
├── outputs/
│ └── final_customer_dataset.csv # merged dataset (predictions + explanations + segments)
├── eda_charts/
├── shap_charts/
├── segmentation_charts/
└── README.md


---

## Running Locally

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Run the pipeline** (in order)
```bash
python src/clean_data.py
python src/eda.py
python src/train_model.py
python src/explain_model.py
python src/segment_customers.py
```

**3. Start the backend**
```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

**4. Serve the frontend**
```bash
cd frontend
python -m http.server 5500
```

Open `http://localhost:5500` in your browser.

---

## Methodology Notes

- **Churn definition**: Customers are labeled as churned based on a feature-driven probability model incorporating satisfaction, engagement, payment history, support activity, and tenure, with added variance to reflect realistic, non-deterministic behavior.
- **Model selection**: Logistic Regression was chosen over Random Forest despite similar complexity, because it achieved higher recall (0.76 vs 0.62) — critical since the cost of missing a churner outweighs the cost of a false positive.
- **Segmentation**: K-Means with k=4, selected via elbow method and silhouette score (0.34) — segments are meaningfully different but moderately overlapping, which is expected given only 3 clustering features.

---

## What This Project Demonstrates

- End-to-end ownership of a data science problem: from raw data to a deployed, usable product
- Moving beyond black-box predictions into genuine explainability (SHAP)
- Translating clustering output into business-actionable customer segments
- Building and connecting a real backend API to a custom front-end (not just a BI tool)
- Deployment and hosting of a full-stack application

---

## Author

**Prajwal Dilip Shevante**
[GitHub](https://github.com/PrajHacks) · [LinkedIn](https://www.linkedin.com/in/prajwalshevante/) · prajwalshevante1@gmail.com
