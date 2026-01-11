"""Data loading and  cleaning.

This module is responsible for reading raw CSVs from data/, standardizing
core data types, handling basic missing values and duplicates, and
constructing a base order-level dataset that can be further enriched by
feature engineering.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from utils import DATA_DIR


RawTables = Dict[str, pd.DataFrame]


def load_raw_olist_data() -> RawTables:
    """Load all Olist CSV files from the data/ directory into DataFrames.

    Returns a dictionary keyed by logical table name (e.g., "customers").
    """

    paths = {
        "customers": DATA_DIR / "olist_customers_dataset.csv",
        "geolocation": DATA_DIR / "olist_geolocation_dataset.csv",
        "orders": DATA_DIR / "olist_orders_dataset.csv",
        "order_items": DATA_DIR / "olist_order_items_dataset.csv",
        "payments": DATA_DIR / "olist_order_payments_dataset.csv",
        "reviews": DATA_DIR / "olist_order_reviews_dataset.csv",
        "products": DATA_DIR / "olist_products_dataset.csv",
        "sellers": DATA_DIR / "olist_sellers_dataset.csv",
        "category_translation": DATA_DIR / "product_category_name_translation.csv",
    }

    tables: RawTables = {}
    for name, path in paths.items():
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"Table {name} loaded from {path} is empty.")
        tables[name] = df

    return tables


def standardize_types_and_clean(tables: RawTables) -> RawTables:
    """Standardize column dtypes and perform light cleaning.

    - Convert timestamp columns to datetime
    - Convert monetary fields to numeric
    - Cast identifier columns to category (memory-friendly)
    - Drop obvious duplicates on primary-esque keys
    """

    customers = tables["customers"].copy()
    geolocation = tables["geolocation"].copy()
    orders = tables["orders"].copy()
    order_items = tables["order_items"].copy()
    payments = tables["payments"].copy()
    reviews = tables["reviews"].copy()
    products = tables["products"].copy()
    sellers = tables["sellers"].copy()
    category_translation = tables["category_translation"].copy()

    # Timestamp conversions
    date_cols_orders = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_cols_orders:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")

    order_items["shipping_limit_date"] = pd.to_datetime(
        order_items["shipping_limit_date"], errors="coerce"
    )
    reviews["review_creation_date"] = pd.to_datetime(
        reviews["review_creation_date"], errors="coerce"
    )
    reviews["review_answer_timestamp"] = pd.to_datetime(
        reviews["review_answer_timestamp"], errors="coerce"
    )

    # Monetary columns
    for col in ("price", "freight_value"):
        order_items[col] = pd.to_numeric(order_items[col], errors="coerce")
    payments["payment_value"] = pd.to_numeric(payments["payment_value"], errors="coerce")

    # Identifier-like columns to category for memory
    for df, cols in [
        (customers, ["customer_id", "customer_unique_id", "customer_state"]),
        (orders, ["order_id", "customer_id", "order_status"]),
        (order_items, ["order_id", "product_id", "seller_id"]),
        (products, ["product_id", "product_category_name"]),
        (sellers, ["seller_id", "seller_state"]),
    ]:
        for c in cols:
            if c in df.columns:
                df[c] = df[c].astype("category")

    # Drop duplicates on primary-key-like columns
    customers = customers.drop_duplicates(subset=["customer_id"])
    orders = orders.drop_duplicates(subset=["order_id"])
    products = products.drop_duplicates(subset=["product_id"])
    sellers = sellers.drop_duplicates(subset=["seller_id"])

    # For reviews, keep latest answer per order
    if "review_answer_timestamp" in reviews.columns:
        reviews = reviews.sort_values("review_answer_timestamp").drop_duplicates(
            subset=["order_id"], keep="last"
        )

    # Basic key presence checks
    orders = orders.dropna(subset=["order_id", "customer_id", "order_purchase_timestamp"])
    order_items = order_items.dropna(subset=["order_id", "product_id", "seller_id"])

    return {
        "customers": customers,
        "geolocation": geolocation,
        "orders": orders,
        "order_items": order_items,
        "payments": payments,
        "reviews": reviews,
        "products": products,
        "sellers": sellers,
        "category_translation": category_translation,
    }


def enrich_products_with_translation(products: pd.DataFrame, category_translation: pd.DataFrame) -> pd.DataFrame:
    """Attach human-readable English category names to the products table."""

    cat = category_translation.rename(
        columns={
            "product_category_name": "product_category_name",
            "product_category_name_english": "product_category_name_english",
        }
    )

    merged = products.merge(cat, on="product_category_name", how="left")
    merged["product_category_name_english"] = merged[
        "product_category_name_english"
    ].fillna("other")
    return merged


def build_base_order_dataset(tables: RawTables) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a base order-level DataFrame and an extended order_items table.

    Returns
    -------
    orders_full : pd.DataFrame
        Order-level table containing orders, customers, aggregated monetary
        values, payments, and reviews.
    order_items_ext : pd.DataFrame
        Item-level table joined with products and sellers, used later for
        distance and product/seller-level aggregations.
    """

    customers = tables["customers"]
    geolocation = tables["geolocation"]  # kept for later distance features
    orders = tables["orders"]
    order_items = tables["order_items"]
    payments = tables["payments"]
    reviews = tables["reviews"]
    products = tables["products"]
    sellers = tables["sellers"]

    # Aggregate order_items at order level
    order_items_agg = (
        order_items.groupby("order_id").agg(
            n_items=("order_item_id", "count"),
            items_price_sum=("price", "sum"),
            freight_value_sum=("freight_value", "sum"),
        )
    ).reset_index()

    # Aggregate payments at order level
    payments_agg = (
        payments.groupby("order_id").agg(
            payment_value_sum=("payment_value", "sum"),
            payment_type_mode=(
                "payment_type",
                lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
            ),
            payment_installments_max=("payment_installments", "max"),
        )
    ).reset_index()

    # Aggregate reviews at order level
    reviews_agg = (
        reviews.groupby("order_id").agg(
            review_score_latest=("review_score", "last"),
            review_creation_date_max=("review_creation_date", "max"),
            review_answer_timestamp_max=("review_answer_timestamp", "max"),
        )
    ).reset_index()

    # Merge orders with customers
    orders_customers = orders.merge(
        customers, on="customer_id", how="left", suffixes=("", "_customer")
    )

    # Merge with aggregates
    orders_items = orders_customers.merge(order_items_agg, on="order_id", how="left")
    orders_items_pay = orders_items.merge(payments_agg, on="order_id", how="left")
    orders_full = orders_items_pay.merge(reviews_agg, on="order_id", how="left")

    # Extended order_items table with product and seller attributes
    order_items_ext = order_items.merge(products, on="product_id", how="left").merge(
        sellers, on="seller_id", how="left", suffixes=("_product", "_seller")
    )

    return orders_full, order_items_ext
