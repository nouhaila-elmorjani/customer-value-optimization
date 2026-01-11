# Customer Retention & Lifetime Value Optimization

**End-to-end e-commerce analytics: delivery, reviews, and customer lifetime value with actionable insights and interactive dashboard.**

---

## Overview

This project analyzes Brazilian Olist e-commerce data to understand customer behavior, optimize retention strategies, and predict lifetime value (CLV). It provides an **end-to-end data science workflow**:

* Clean, enriched order-level data
* Feature engineering capturing business insights (time, delivery, monetary, geography, interactions)
* Predictive models for:

  * Delivery delays
  * Review scores (multiclass and binary satisfaction)
  * Customer lifetime value (top 5% high-value customers)
* Interactive **Plotly Dash dashboard** for stakeholders to explore KPIs, customer segments, and delivery performance

**Key skills & highlights:** Python, Pandas, Scikit-learn, CatBoost, XGBoost, Plotly Dash, feature engineering, regression & classification, class imbalance handling, threshold tuning.

---

## Project Structure

```
data/                 # Raw Olist CSVs
notebooks/
  olist_analysis.ipynb # Main analysis: EDA, feature engineering, modeling, KPIs, exports
src/
  data_loading.py       # Load & clean raw tables
  feature_engineering.py # Time, delivery, monetary, CLV, reviews, distance, interaction features
  eda.py                # EDA plotting utilities
  modeling.py           # Modeling utilities for delivery delay, review score, binary satisfaction, CLV
  utils.py              # Paths, seeding, save functions
  dashboard.py          # Plotly Dash interactive dashboard
outputs/
  enriched_orders.csv   # Final feature-rich dataset
  top10_categories_revenue.csv
  top10_sellers_revenue.csv
requirements.txt       # Python dependencies
```

> Note: Figures are saved in `figures/` but optional for GitHub; key visuals are included below.

---

## Modeling Tasks & Results

### 1. Delivery Delay Regression

**Target:** `delivery_delay_days` (days late)
**Models & Metrics:**

| Model            | RMSE | MAE  | R²    |
| ---------------- | ---- | ---- | ----- |
| RandomForest     | 8.56 | 5.05 | 0.356 |
| GradientBoosting | 8.54 | 4.97 | 0.359 |
| XGBoost          | 8.45 | 4.86 | 0.372 |

> Models capture key patterns of delays; longer distances and estimated delivery times drive higher errors.

---

### 2. Multiclass Review Score Prediction

**Target:** `review_score_latest` (1–5)

| Model                      | Accuracy | F1 Macro | F1 Weighted |
| -------------------------- | -------- | -------- | ----------- |
| GradientBoostingClassifier | 0.453    | 0.311    | 0.471       |
| CatBoostClassifier         | 0.411    | 0.299    | 0.441       |

---

### 3. Binary Customer Satisfaction

**Target:** `review_binary` = 1 if review ≥ 4

| Model                      | Accuracy | ROC AUC | Macro F1 | Weighted F1 | Positive F1 | Best Threshold |
| -------------------------- | -------- | ------- | -------- | ----------- | ----------- | -------------- |
| GradientBoostingClassifier | 0.780    | 0.715   | 0.681    | 0.775       | 0.858       | 0.15           |
| CatBoostClassifier         | 0.779    | 0.716   | 0.680    | 0.774       | 0.858       | 0.15           |

> Threshold tuning optimizes F1 score for actionable classification of satisfied vs dissatisfied customers.

---

### 4. CLV Regression (Top 5% Customers)

**Target:** `customer_total_revenue`

| Model                     | RMSE  | MAE  |
| ------------------------- | ----- | ---- |
| GradientBoostingRegressor | 21.04 | 3.80 |

> Focused on high-value customers to prioritize retention and loyalty programs.

---

## Key Insights

* **Revenue & demand:**
  Concentrated among a few product categories and top sellers; strong seasonality. Top 5% of customers contribute disproportionately to GMV.

* **Delivery performance:**
  Late deliveries increase with distance and longer estimated times; certain states have higher prediction errors.

* **Customer satisfaction:**
  Late deliveries lower review scores; payment type and order value also affect satisfaction slightly.

* **High-value customers:**
  Targeted retention campaigns (e.g., perks, faster shipping, follow-up after low reviews) have high ROI potential.

---

## Dashboard

Interactive **Plotly Dash** dashboard allows business users to explore:

* KPIs: total revenue, avg review score, late delivery rate
* Orders and revenue trends over time
* Top product categories & top sellers by revenue
* Delivery delay distributions and performance by state
* High-value customers and CLV predictions

**Screenshots / Key Figures:**

**CLV Regression (Top 5% Customers)**
<!-- Uploading "clv_actual_vs_predicted.png"... -->
![alt text](image.png)

**Top Product Categories by Revenue**
![Top Categories](figures/top_categories_revenue.png)

![alt text](image-1.png)

> To run locally:

```bash
cd src
python dashboard.py
```

Then open [http://127.0.0.1:8050/](http://127.0.0.1:8050/)

---

## Reproduce Analysis

1. Clone repo and navigate into project.
2. Create & activate virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run `notebooks/olist_analysis.ipynb` (or all cells up to Section 7) to generate features, models, outputs.
5. Figures (optional) and enriched datasets will be saved to `figures/` and `outputs/`.

---


