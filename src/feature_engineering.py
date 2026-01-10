from typing import Tuple, Dict

import numpy as np
import pandas as pd


def add_time_features(orders: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features derived from the purchase timestamp."""

    orders = orders.copy()
    ts = orders["order_purchase_timestamp"]
    orders["purchase_year"] = ts.dt.year
    orders["purchase_month"] = ts.dt.month
    orders["purchase_day"] = ts.dt.day
    orders["purchase_weekday"] = ts.dt.weekday
    orders["purchase_hour"] = ts.dt.hour
    orders["is_weekend"] = orders["purchase_weekday"].isin([5, 6])

    # Helper column for time-series aggregations
    orders["order_purchase_date"] = ts.dt.to_period("M").dt.to_timestamp()
    return orders


def add_binary_review_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add a binary customer satisfaction target based on review scores.

    review_binary = 1 if review_score >= 4 else 0

    Uses the latest review score per order/customer where available.
    """

    df = df.copy()
    # Prefer the engineered latest review score if present, otherwise fall back.
    score_col = "review_score_latest" if "review_score_latest" in df.columns else "review_score"
    df["review_binary"] = (df[score_col] >= 4).astype(int)
    return df


def add_delivery_features(orders: pd.DataFrame) -> pd.DataFrame:
    """Create delivery-related features such as delay and total time."""

    orders = orders.copy()
    purchase = orders["order_purchase_timestamp"]
    delivered = orders["order_delivered_customer_date"]
    estimated = orders["order_estimated_delivery_date"]

    orders["delivery_time_days"] = (delivered - purchase).dt.total_seconds() / 86400
    orders["delivery_estimated_days"] = (
        (estimated - purchase).dt.total_seconds() / 86400
    )
    orders["delivery_delay_days"] = (
        (delivered - estimated).dt.total_seconds() / 86400
    )
    orders["is_late"] = orders["delivery_delay_days"] > 0
    return orders


def add_monetary_features(orders: pd.DataFrame) -> pd.DataFrame:
    """Add order-level monetary aggregations such as total value and avg item price."""

    orders = orders.copy()
    orders["items_price_sum"] = orders["items_price_sum"].fillna(0)
    orders["freight_value_sum"] = orders["freight_value_sum"].fillna(0)
    orders["total_order_value"] = orders["items_price_sum"] + orders["freight_value_sum"]
    orders["avg_item_price"] = orders["items_price_sum"] / orders["n_items"]
    return orders


def add_customer_clv_features(orders: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute CLV-style aggregates at customer level and attach to orders.

    Returns the enriched orders and the standalone customer-level table.
    """

    orders = orders.copy()
    clv = (
        orders.groupby("customer_id", observed=False).agg(
            customer_n_orders=("order_id", "nunique"),
            customer_total_revenue=("total_order_value", "sum"),
            customer_avg_order_value=("total_order_value", "mean"),
            customer_last_purchase=("order_purchase_timestamp", "max"),
        )
    ).reset_index()

    orders = orders.merge(clv, on="customer_id", how="left")
    return orders, clv


def add_review_aggregates(
    orders: pd.DataFrame, order_items_ext: pd.DataFrame, reviews: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add review-based aggregates for products, sellers, and customers.

    Returns enriched orders, product-level, and seller-level review tables.
    """

    orders = orders.copy()

    prod_reviews = order_items_ext.merge(
        reviews[["order_id", "review_score"]], on="order_id", how="left"
    )
    prod_reviews_agg = (
        prod_reviews.groupby("product_id", observed=False).agg(
            product_review_score_mean=("review_score", "mean"),
            product_review_count=("review_score", "count"),
        )
    ).reset_index()

    seller_reviews_agg = (
        prod_reviews.groupby("seller_id", observed=False).agg(
            seller_review_score_mean=("review_score", "mean"),
            seller_review_count=("review_score", "count"),
        )
    ).reset_index()

    cust_reviews_agg = (
        orders.groupby("customer_id", observed=False).agg(
            customer_review_score_mean=("review_score_latest", "mean"),
            customer_review_count=("review_score_latest", "count"),
        )
    ).reset_index()

    orders = orders.merge(cust_reviews_agg, on="customer_id", how="left")
    return orders, prod_reviews_agg, seller_reviews_agg


def add_geolocation_distance(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    sellers: pd.DataFrame,
    geolocation: pd.DataFrame,
    order_items_ext: pd.DataFrame,
) -> pd.DataFrame:
    """Compute haversine distance between customer and seller and aggregate to order level."""

    orders = orders.copy()

    geo_zip = geolocation.groupby("geolocation_zip_code_prefix").agg(
        geo_lat=("geolocation_lat", "mean"),
        geo_lng=("geolocation_lng", "mean"),
    ).reset_index()

    customers_geo = customers.merge(
        geo_zip,
        left_on="customer_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    ).rename(columns={"geo_lat": "customer_lat", "geo_lng": "customer_lng"})

    sellers_geo = sellers.merge(
        geo_zip,
        left_on="seller_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    ).rename(columns={"geo_lat": "seller_lat", "geo_lng": "seller_lng"})

    # Attach customer_id to order_items_ext via orders so we can join to customer geography
    order_items_with_customer = order_items_ext.merge(
        orders[["order_id", "customer_id"]],
        on="order_id",
        how="left",
    )

    order_items_geo = order_items_with_customer.merge(
        customers_geo[["customer_id", "customer_lat", "customer_lng"]],
        on="customer_id",
        how="left",
    ).merge(
        sellers_geo[["seller_id", "seller_lat", "seller_lng"]],
        on="seller_id",
        how="left",
    )

    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    order_items_geo["distance_km"] = haversine(
        order_items_geo["customer_lat"],
        order_items_geo["customer_lng"],
        order_items_geo["seller_lat"],
        order_items_geo["seller_lng"],
    )

    distance_agg = (
        order_items_geo.groupby("order_id").agg(distance_km_mean=("distance_km", "mean"))
    ).reset_index()

    orders = orders.merge(distance_agg, on="order_id", how="left")
    return orders


def add_interaction_features(orders: pd.DataFrame) -> pd.DataFrame:
    """Create simple interaction features for modeling.

    Examples
    --------
    - freight_to_price_ratio: freight_value_sum / items_price_sum
    - delay_x_total_value: delivery_delay_days * total_order_value
    """

    orders = orders.copy()

    if {"freight_value_sum", "items_price_sum"}.issubset(orders.columns):
        denom = orders["items_price_sum"].replace(0, np.nan)
        ratio = orders["freight_value_sum"] / denom
        ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(0)
        orders["freight_to_price_ratio"] = ratio

    if {"delivery_delay_days", "total_order_value"}.issubset(orders.columns):
        orders["delay_x_total_value"] = (
            orders["delivery_delay_days"] * orders["total_order_value"]
        )

    return orders


def build_model_feature_sets(orders: pd.DataFrame) -> Dict[str, dict]:
    """Define feature/target sets for each modeling task.

    Returns a dict with configuration for delay, review, and CLV models.
    """

    cfg = {}

    cfg["delay"] = {
        "target": "delivery_delay_days",
        "numeric": [
            "items_price_sum",
            "freight_value_sum",
            "payment_value_sum",
            "n_items",
            "total_order_value",
            "distance_km_mean",
            "delivery_estimated_days",
            "purchase_month",
            "purchase_hour",
            "freight_to_price_ratio",
        ],
        "categorical": ["payment_type_mode", "customer_state", "is_weekend"],
    }

    cfg["review"] = {
        "target": "review_score_latest",
        "numeric": [
            "total_order_value",
            "n_items",
            "delivery_delay_days",
            "distance_km_mean",
            "customer_total_revenue",
            "customer_n_orders",
            "freight_to_price_ratio",
            "delay_x_total_value",
        ],
        "categorical": ["payment_type_mode", "customer_state"],
    }

    return cfg
