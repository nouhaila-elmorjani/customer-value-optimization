"""Exploratory Data Analysis (EDA) utilities for the Olist dataset.

These functions encapsulate common plots so they can be reused from
notebooks or scripts while keeping the notebook focused on narrative.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import save_figure


def plot_missingness(orders_full: pd.DataFrame, top_n: int = 20) -> None:
    """Plot the top-n columns by percent missingness."""

    nulls = orders_full.isna().sum().to_frame("n_missing")
    nulls["pct_missing"] = (nulls["n_missing"] / len(orders_full) * 100).round(2)
    nulls_sorted = nulls.sort_values("pct_missing", ascending=False).head(top_n)

    plt.figure()
    nulls_sorted["pct_missing"].plot(kind="bar")
    plt.ylabel("Percent missing (%)")
    plt.title("Top Columns by Missingness (Order-Level Dataset)")
    save_figure("eda_missingness_top")
    plt.show()


def plot_univariate_distributions(orders_full: pd.DataFrame) -> None:
    """Plot key univariate distributions for monetary and satisfaction metrics."""

    plt.figure()
    sns.histplot(orders_full["items_price_sum"].dropna(), bins=50)
    plt.title("Distribution of Items Price Sum per Order")
    plt.xlabel("Items price sum (BRL)")
    save_figure("dist_items_price_sum")
    plt.show()

    plt.figure()
    sns.histplot(orders_full["freight_value_sum"].dropna(), bins=50)
    plt.title("Distribution of Freight Value per Order")
    plt.xlabel("Freight value (BRL)")
    save_figure("dist_freight_value_sum")
    plt.show()

    if "review_score_latest" in orders_full.columns:
        plt.figure()
        sns.countplot(x="review_score_latest", data=orders_full)
        plt.title("Review Score Distribution")
        save_figure("dist_review_scores")
        plt.show()


def plot_time_series(ts_orders: pd.DataFrame) -> None:
    """Plot orders and revenue over time (monthly)."""

    plt.figure()
    sns.lineplot(x="order_purchase_date", y="n_orders", data=ts_orders)
    plt.title("Orders per Month")
    plt.xlabel("Month")
    plt.ylabel("Number of orders")
    save_figure("orders_per_month")
    plt.show()

    plt.figure()
    sns.lineplot(x="order_purchase_date", y="revenue", data=ts_orders)
    plt.title("Revenue per Month")
    plt.xlabel("Month")
    plt.ylabel("Revenue (BRL)")
    save_figure("revenue_per_month")
    plt.show()


def plot_category_and_seller_performance(
    cat_perf: pd.DataFrame, seller_perf: pd.DataFrame
) -> None:
    """Plot top categories and sellers by revenue."""

    if not cat_perf.empty:
        plt.figure()
        sns.barplot(x="revenue", y="product_category_name_english", data=cat_perf)
        plt.title("Top Categories by Revenue")
        plt.xlabel("Revenue (BRL)")
        save_figure("top_categories_revenue")
        plt.show()

    if not seller_perf.empty:
        plt.figure()
        sns.barplot(x="revenue", y="seller_id", data=seller_perf)
        plt.title("Top Sellers by Revenue")
        plt.xlabel("Revenue (BRL)")
        save_figure("top_sellers_revenue")
        plt.show()


def plot_geography_and_payments(customers: pd.DataFrame, payments: pd.DataFrame, order_items: pd.DataFrame) -> None:
    """Plot customer geography, freight vs price, and payment type distributions."""

    cust_state_counts = customers["customer_state"].value_counts().reset_index()
    cust_state_counts.columns = ["customer_state", "n_customers"]

    plt.figure()
    sns.barplot(x="customer_state", y="n_customers", data=cust_state_counts)
    plt.xticks(rotation=90)
    plt.title("Customer Count by State")
    save_figure("customers_by_state")
    plt.show()

    plt.figure()
    sample = order_items.sample(min(20000, len(order_items)), random_state=0)
    sns.scatterplot(x="price", y="freight_value", data=sample, alpha=0.3)
    plt.title("Freight Value vs Item Price (Sample)")
    plt.xlabel("Item price (BRL)")
    plt.ylabel("Freight value (BRL)")
    save_figure("freight_vs_price_scatter")
    plt.show()

    pay_type_counts = payments["payment_type"].value_counts().reset_index()
    pay_type_counts.columns = ["payment_type", "n_payments"]

    plt.figure()
    sns.barplot(x="payment_type", y="n_payments", data=pay_type_counts)
    plt.title("Payment Type Distribution")
    save_figure("payment_type_distribution")
    plt.show()


def plot_delivery_and_correlations(orders_full: pd.DataFrame) -> None:
    """Plot delivery delay distributions and a correlation heatmap."""

    plt.figure()
    sns.histplot(orders_full["delivery_delay_days"].dropna(), bins=50)
    plt.title("Distribution of Delivery Delay (Days)")
    plt.xlabel("Delivery delay (days, positive = late)")
    save_figure("delivery_delay_distribution")
    plt.show()

    if "review_score_latest" in orders_full.columns:
        plt.figure()
        sns.boxplot(x="review_score_latest", y="delivery_delay_days", data=orders_full)
        plt.title("Delivery Delay by Review Score")
        save_figure("delay_by_review_score")
        plt.show()

    num_cols = [
        "items_price_sum",
        "freight_value_sum",
        "payment_value_sum",
        "delivery_time_days",
        "delivery_estimated_days",
        "delivery_delay_days",
        "n_items",
        "review_score_latest",
    ]
    num_cols = [c for c in num_cols if c in orders_full.columns]
    if num_cols:
        corr = orders_full[num_cols].corr()
        plt.figure()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Heatmap (Key Numeric Features)")
        save_figure("correlation_heatmap_key_features")
        plt.show()

        # pairplot for a compact view of bivariate relationships
        pairplot_sample = orders_full[num_cols].dropna().sample(
            min(2000, len(orders_full)), random_state=0
        )
        sns.pairplot(pairplot_sample[num_cols], corner=True)
        plt.suptitle("Pairplot of Key Numeric Features", y=1.02)
        save_figure("pairplot_key_numeric_features")
        plt.show()
