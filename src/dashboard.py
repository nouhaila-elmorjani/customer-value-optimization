"""Plotly Dash application for Olist KPIs and predictive insights.

This app reads the enriched order-level dataset produced by the
analysis notebook (outputs/enriched_orders.csv) and exposes
interactive views for business users:

- Top products, categories, and sellers by revenue
- Geographic view of average delivery delay by customer state
- Top customers by CLV
- Review score distribution and relationship with delivery delay
- Optional simple delivery-delay "prediction" demo for a user-input order

"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

# Project paths -------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

ENRICHED_PATH = OUTPUTS_DIR / "enriched_orders.csv"
TOP10_CATEGORIES_PATH = OUTPUTS_DIR / "top10_categories_revenue.csv"
TOP10_SELLERS_PATH = OUTPUTS_DIR / "top10_sellers_revenue.csv"


def load_enriched_orders(path: Path = ENRICHED_PATH) -> pd.DataFrame:
    """Load the enriched orders CSV produced by the notebook.

    Parameters
    ----------
    path : Path
        Location of the enriched orders CSV.

    Returns
    -------
    pandas.DataFrame
        Enriched order-level table with engineered features.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Expected enriched dataset at {path}, but it was not found. "
            "Run notebooks/olist_analysis.ipynb first to generate it."
        )

    df = pd.read_csv(path, parse_dates=["order_purchase_timestamp"])

    #  sanity clean-up
    if "delivery_delay_days" in df.columns:
        df["delivery_delay_days"] = df["delivery_delay_days"].fillna(0)

    return df


def compute_top_entities(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute top categories, sellers, and customers for KPI views."""

    # Top product categories by revenue
    if TOP10_CATEGORIES_PATH.exists():
        cat_df = pd.read_csv(TOP10_CATEGORIES_PATH)
    else:
        cat_df = pd.DataFrame(columns=["product_category_name_english", "revenue"])

    # Top sellers by revenue
    if TOP10_SELLERS_PATH.exists():
        seller_df = pd.read_csv(TOP10_SELLERS_PATH)
    else:
        seller_df = pd.DataFrame(columns=["seller_id", "revenue"])

    # Top customers by CLV (customer_total_revenue from feature_engineering)
    if {"customer_id", "customer_total_revenue", "customer_n_orders"}.issubset(df.columns):
        cust_df = (
            df.drop_duplicates("customer_id")[
                ["customer_id", "customer_total_revenue", "customer_n_orders"]
            ]
            .sort_values("customer_total_revenue", ascending=False)
            .head(15)
        )
        
        cust_df["customer_label"] = (
            "C" + (cust_df.reset_index().index + 1).astype(str).str.zfill(2)
        )
    else:
        cust_df = pd.DataFrame(
            columns=[
                "customer_id",
                "customer_total_revenue",
                "customer_n_orders",
                "customer_label",
            ]
        )

    return cat_df, seller_df, cust_df


def create_app(df: pd.DataFrame) -> Dash:
    """Create and configure the Dash app.

    Parameters
    ----------
    df : pandas.DataFrame
        Enriched orders dataframe.
    """

    app = Dash(__name__)

    # Prepare common filter options
    state_options: List[dict] = [
        {"label": s, "value": s}
        for s in sorted(df["customer_state"].dropna().unique())
    ] if "customer_state" in df.columns else []

    date_min = df["order_purchase_timestamp"].min() if "order_purchase_timestamp" in df.columns else None
    date_max = df["order_purchase_timestamp"].max() if "order_purchase_timestamp" in df.columns else None

    cat_df, seller_df, cust_df = compute_top_entities(df)

    app.layout = html.Div(
        [
            html.H1("Olist KPIs and Delivery/Review Insights"),
            html.Div(
                [
                    html.Div(
                        [
                            html.H4("Filters"),
                            html.Label("Customer state"),
                            dcc.Dropdown(
                                id="state-filter",
                                options=state_options,
                                value=[],
                                multi=True,
                                placeholder="Select one or more states",
                            ),
                            html.Br(),
                            html.Label("Purchase date range"),
                            dcc.DatePickerRange(
                                id="date-range",
                                min_date_allowed=date_min,
                                max_date_allowed=date_max,
                                start_date=date_min,
                                end_date=date_max,
                            ),
                        ],
                        style={"flex": "1", "padding": "1rem", "borderRight": "1px solid #eee"},
                    ),
                    html.Div(
                        [
                            html.H4("Overview KPIs"),
                            html.Div(id="kpi-summary"),
                            html.Br(),
                            html.Div(
                                [
                                    dcc.Graph(id="top-categories-fig"),
                                    dcc.Graph(id="top-sellers-fig"),
                                ],
                                style={"display": "flex", "flexWrap": "wrap"},
                            ),
                            html.Div(
                                [
                                    dcc.Graph(id="delay-by-state-fig"),
                                    dcc.Graph(id="top-customers-fig"),
                                ],
                                style={"display": "flex", "flexWrap": "wrap"},
                            ),
                            html.Div(
                                [
                                    dcc.Graph(id="review-distribution-fig"),
                                    dcc.Graph(id="delay-vs-review-fig"),
                                ],
                                style={"display": "flex", "flexWrap": "wrap"},
                            ),
                        ],
                        style={"flex": "3", "padding": "1rem"},
                    ),
                ],
                style={"display": "flex"},
            ),
        ]
    )

    # Callbacks ------------------------------------------------------------

    @app.callback(
        [
            Output("kpi-summary", "children"),
            Output("top-categories-fig", "figure"),
            Output("top-sellers-fig", "figure"),
            Output("delay-by-state-fig", "figure"),
            Output("top-customers-fig", "figure"),
            Output("review-distribution-fig", "figure"),
            Output("delay-vs-review-fig", "figure"),
        ],
        [
            Input("state-filter", "value"),
            Input("date-range", "start_date"),
            Input("date-range", "end_date"),
        ],
    )
    def update_kpis(states, start_date, end_date):  # type: ignore[override]
        # Filter data according to selections
        dff = df.copy()

        if states:
            dff = dff[dff["customer_state"].isin(states)]
        if start_date and end_date and "order_purchase_timestamp" in dff.columns:
            mask = (dff["order_purchase_timestamp"] >= start_date) & (
                dff["order_purchase_timestamp"] <= end_date
            )
            dff = dff[mask]

        # KPIs
        total_revenue = dff.get("total_order_value", pd.Series(dtype=float)).sum()
        n_orders = dff.get("order_id", pd.Series(dtype=object)).nunique()
        avg_delay = dff.get("delivery_delay_days", pd.Series(dtype=float)).mean()
        avg_review = dff.get("review_score_latest", pd.Series(dtype=float)).mean()

        kpi_children = html.Div(
            [
                html.Div(
                    [
                        html.H5("Total revenue (BRL)"),
                        html.P(f"{total_revenue:,.0f}" if not np.isnan(total_revenue) else "-"),
                    ],
                    style={"display": "inline-block", "marginRight": "2rem"},
                ),
                html.Div(
                    [
                        html.H5("Number of orders"),
                        html.P(f"{n_orders:,}"),
                    ],
                    style={"display": "inline-block", "marginRight": "2rem"},
                ),
                html.Div(
                    [
                        html.H5("Average delay (days)"),
                        html.P(f"{avg_delay:.2f}" if avg_delay == avg_delay else "-"),
                    ],
                    style={"display": "inline-block", "marginRight": "2rem"},
                ),
                html.Div(
                    [
                        html.H5("Average review score"),
                        html.P(f"{avg_review:.2f}" if avg_review == avg_review else "-"),
                    ],
                    style={"display": "inline-block"},
                ),
            ]
        )

        # Top categories & sellers
        cat_df_f, seller_df_f, cust_df_f = compute_top_entities(dff)

        top_cat_fig = px.bar(
            cat_df_f,
            x="product_category_name_english",
            y="revenue",
            title="Top Product Categories by Revenue",
            color_discrete_sequence=["#2E86AB"],
        )
        top_cat_fig.update_layout(
            template="plotly_white",
            xaxis_title="Category",
            yaxis_title="Revenue (BRL)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        top_seller_fig = px.bar(
            seller_df_f,
            x="seller_id",
            y="revenue",
            title="Top Sellers by Revenue",
            color_discrete_sequence=["#F39C12"],
        )
        top_seller_fig.update_layout(
            template="plotly_white",
            xaxis_title="Seller",
            yaxis_title="Revenue (BRL)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        # Delay by state (bar chart)
        if {"customer_state", "delivery_delay_days"}.issubset(dff.columns):
            delay_state = (
                dff.groupby("customer_state", observed=False)["delivery_delay_days"]
                .mean()
                .reset_index(name="avg_delay")
                .sort_values("avg_delay", ascending=False)
            )
            delay_state_fig = px.bar(
                delay_state,
                x="customer_state",
                y="avg_delay",
                title="Average Delivery Delay by State",
                color_discrete_sequence=["#E74C3C"],
            )
            delay_state_fig.update_layout(
                template="plotly_white",
                xaxis_title="State",
                yaxis_title="Average delay (days)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
        else:
            delay_state_fig = px.bar(title="Average Delivery Delay by State")

        # Top customers by CLV
        top_customers_fig = px.bar(
            cust_df_f,
            x="customer_label",
            y="customer_total_revenue",
            hover_data=["customer_id", "customer_n_orders"],
            title="Top Customers by CLV",
            color_discrete_sequence=["#27AE60"],
        )
        top_customers_fig.update_layout(
            template="plotly_white",
            xaxis_title="Customer (anonymized)",
            yaxis_title="Total revenue (BRL)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        # Review score distribution
        if "review_score_latest" in dff.columns:
            review_dist_fig = px.histogram(
                dff,
                x="review_score_latest",
                nbins=5,
                title="Review Score Distribution",
            )
            review_dist_fig.update_traces(marker_color="#8E44AD")
            review_dist_fig.update_layout(
                template="plotly_white",
                xaxis_title="Review score",
                yaxis_title="Count",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
        else:
            review_dist_fig = px.histogram(title="Review Score Distribution")

        # Delay vs review correlation
        if {"delivery_delay_days", "review_score_latest"}.issubset(dff.columns):
            delay_vs_review_fig = px.scatter(
                dff,
                x="delivery_delay_days",
                y="review_score_latest",
                title="Delivery Delay vs Review Score",
                opacity=0.3,
            )
            delay_vs_review_fig.update_traces(marker_color="#16A085")
            delay_vs_review_fig.update_layout(
                template="plotly_white",
                xaxis_title="Delivery delay (days)",
                yaxis_title="Review score",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
        else:
            delay_vs_review_fig = px.scatter(title="Delivery Delay vs Review Score")

        return (
            kpi_children,
            top_cat_fig,
            top_seller_fig,
            delay_state_fig,
            top_customers_fig,
            review_dist_fig,
            delay_vs_review_fig,
        )

    return app


def main() -> None:
    """Entry point for running the Dash app.

    Example
    -------
    From the project root::

        python -m src.dashboard

    Ensure that outputs/enriched_orders.csv exists (run the notebook first).
    """

    df = load_enriched_orders()
    app = create_app(df)
    
    app.run(debug=True)


if __name__ == "__main__": 
    main()
