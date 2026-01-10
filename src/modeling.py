"""Modeling utilities for Olist delivery delay, review scores, and CLV."""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
)
try:
    from xgboost import XGBRegressor  

    HAS_XGBOOST = True
except Exception:  
    XGBRegressor = None  
    HAS_XGBOOST = False

try:
    from catboost import CatBoostClassifier  

    HAS_CATBOOST = True
except Exception: 
    CatBoostClassifier = None  
    HAS_CATBOOST = False

from utils import RANDOM_SEED, save_figure


def _find_best_threshold(y_true, y_proba, pos_label: int = 1, thresholds=None) -> Dict[str, float]:
    """Grid-search a probability threshold that maximizes F1 for the
    positive class.

    Parameters
    ----------
    y_true: array-like
        Ground-truth binary labels.
    y_proba: array-like
        Predicted probabilities for the positive class.
    pos_label: int, default 1
        Label to treat as the positive class.
    thresholds: iterable of float, optional
        Custom list of thresholds to evaluate. If None, a default grid
        from 0.1 to 0.9 is used.
    """

    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)

    best = {"threshold": 0.5, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    for thr in thresholds:
        preds = (y_proba >= thr).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            preds,
            pos_label=pos_label,
            average="binary",
            zero_division=0,
        )
        if f1 > best["f1"]:
            best.update(
                {
                    "threshold": float(thr),
                    "f1": float(f1),
                    "precision": float(precision),
                    "recall": float(recall),
                }
            )

    return best


def _plot_threshold_curve(
    y_true, y_proba, model_name: str, pos_label: int = 1, thresholds=None
) -> None:
    """Plot precision, recall, and F1-score as a function of threshold.

    This is primarily for diagnostics and portfolio-ready storytelling
    around how threshold tuning trades off precision vs. recall for the
    positive (dissatisfied) class.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)

    f1_scores = []
    precisions = []
    recalls = []

    for thr in thresholds:
        preds = (y_proba >= thr).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            preds,
            pos_label=pos_label,
            average="binary",
            zero_division=0,
        )
        precisions.append(float(precision))
        recalls.append(float(recall))
        f1_scores.append(float(f1))

    plt.figure()
    sns.lineplot(x=thresholds, y=f1_scores, label="F1-score")
    sns.lineplot(x=thresholds, y=precisions, label="Precision", linestyle="--")
    sns.lineplot(x=thresholds, y=recalls, label="Recall", linestyle="--")
    plt.xlabel("Probability threshold")
    plt.ylabel("Score")
    plt.title(f"{model_name}: Precision/Recall/F1 vs Threshold")
    plt.legend()
    save_figure(f"{model_name.lower().replace(' ', '_')}_threshold_tuning")
    plt.show()


def _evaluate_regression(name: str, model, X_train, X_test, y_train, y_test) -> Dict[str, float]:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure()
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.3)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{name}: Actual vs Predicted")
    save_figure(f"{name.lower().replace(' ', '_')}_actual_vs_pred")
    plt.show()

    return {"rmse": rmse, "mae": mae, "r2": r2}


def build_delay_dataset(
    orders: pd.DataFrame,
    numeric_features: list,
    categorical_features: list,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """Prepare train/test sets and a preprocessor for delay regression."""

    delay_df = orders.dropna(subset=["delivery_delay_days"]).copy()
    y = delay_df["delivery_delay_days"]
    X = delay_df[numeric_features + categorical_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return X_train, X_test, y_train, y_test, preprocessor


def train_delay_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    preprocessor: ColumnTransformer,
) -> Tuple[Dict[str, dict], Dict[str, Pipeline]]:
    """Train Random Forest, Gradient Boosting, and XGBoost regressors."""

    models = {}
    metrics = {}

    rf = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=150,
                    max_depth=None,
                    random_state=RANDOM_SEED,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    metrics["RandomForest"] = _evaluate_regression(
        "Random Forest Delay", rf, X_train, X_test, y_train, y_test
    )
    models["RandomForest"] = rf

    gbr = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", GradientBoostingRegressor(random_state=RANDOM_SEED)),
        ]
    )
    metrics["GradientBoosting"] = _evaluate_regression(
        "Gradient Boosting Delay", gbr, X_train, X_test, y_train, y_test
    )
    models["GradientBoosting"] = gbr

    if HAS_XGBOOST:
        xgb = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    XGBRegressor(
                        random_state=RANDOM_SEED,
                        n_estimators=200,
                        learning_rate=0.1,
                        max_depth=6,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="reg:squarederror",
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        metrics["XGBoost"] = _evaluate_regression(
            "XGBoost Delay", xgb, X_train, X_test, y_train, y_test
        )
        models["XGBoost"] = xgb

    return metrics, models


def build_review_dataset(
    orders: pd.DataFrame,
    numeric_features: list,
    categorical_features: list,
):
    """Prepare train/test sets for review score classification (1–5).

    Target column is ``review_score``; if absent, falls back to
    ``review_score_latest`` and maps it into ``review_score``.
    """

    df = orders.copy()

    # Ensure we have a "review_score" column to use as target
    if "review_score" not in df.columns:
        if "review_score_latest" in df.columns:
            df["review_score"] = df["review_score_latest"]
        else:
            raise KeyError("Neither 'review_score' nor 'review_score_latest' found in dataframe")

    df = df.dropna(subset=["review_score"]).copy()
    df["review_score"] = df["review_score"].astype(int)

    y = df["review_score"]
    X = df[numeric_features + categorical_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return X_train, X_test, y_train, y_test, preprocessor, X, y


def train_review_models(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
    X_all,
    y_all,
) -> Dict[str, dict]:
    """Train review score (1–5) classifiers and return metrics.

    Returns a dict keyed by model name containing accuracy,
    classification_report, and confusion_matrix.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    metrics: Dict[str, dict] = {}

    # Compute sample weights to handle class imbalance in the multiclass
    # review score distribution.
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    # Always train Gradient Boosting as a strong baseline / fallback
    gbc = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", GradientBoostingClassifier(random_state=RANDOM_SEED)),
        ]
    )
    gbc.fit(X_train, y_train, model__sample_weight=sample_weight)
    y_pred_gbc = gbc.predict(X_test)
    acc_gbc = float(accuracy_score(y_test, y_pred_gbc))
    report_gbc = classification_report(y_test, y_pred_gbc, output_dict=True)
    cm_gbc = confusion_matrix(y_test, y_pred_gbc)

    plt.figure()
    sns.heatmap(cm_gbc, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix – Gradient Boosting Review Score")
    save_figure("gbc_review_confusion_matrix")
    plt.show()

    metrics["GradientBoostingClassifier"] = {
        "accuracy": acc_gbc,
        "classification_report": report_gbc,
        "confusion_matrix": cm_gbc,
        "f1_macro": report_gbc["macro avg"]["f1-score"],
        "f1_weighted": report_gbc["weighted avg"]["f1-score"],
    }

    # CatBoost Classifier (work on encoded data) – preferred when available
    if HAS_CATBOOST:
        X_all_encoded = preprocessor.fit_transform(X_all)
        X_train_enc, X_test_enc, y_train_enc, y_test_enc = train_test_split(
            X_all_encoded,
            y_all,
            test_size=0.2,
            random_state=RANDOM_SEED,
            stratify=y_all,
        )

        cat = CatBoostClassifier(
            depth=6,
            learning_rate=0.1,
            loss_function="MultiClass",
            random_seed=RANDOM_SEED,
            verbose=False,
            auto_class_weights="Balanced",
        )
        cat.fit(X_train_enc, y_train_enc)
        y_pred_cat = cat.predict(X_test_enc).ravel().astype(int)

        acc_cat = float(accuracy_score(y_test_enc, y_pred_cat))
        report_cat = classification_report(y_test_enc, y_pred_cat, output_dict=True)
        cm_cat = confusion_matrix(y_test_enc, y_pred_cat)

        plt.figure()
        sns.heatmap(cm_cat, annot=True, fmt="d", cmap="Greens")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix – CatBoost Review Score")
        save_figure("catboost_review_confusion_matrix")
        plt.show()

        metrics["CatBoostClassifier"] = {
            "accuracy": acc_cat,
            "classification_report": report_cat,
            "confusion_matrix": cm_cat,
            "f1_macro": report_cat["macro avg"]["f1-score"],
            "f1_weighted": report_cat["weighted avg"]["f1-score"],
        }

    return metrics


def train_binary_review_models(
    df: pd.DataFrame,
    features: list,
    use_class_weight: bool = False,
    tune_threshold: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """Train binary customer satisfaction models on ``review_binary``.

    Supports optional class balancing via ``use_class_weight`` to improve
    recall for the minority (low-score) class when reviews are skewed
    toward positive scores.
    """
    from sklearn.metrics import classification_report

    X = df[features]
    y = df["review_binary"].astype(int)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Separate numeric and categorical features for proper preprocessing
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    # Optional class balancing to give more weight to minority (low-score) class
    sample_weight = None
    if use_class_weight:
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    results = []

    # 1) Gradient Boosting via sklearn Pipeline (works well with ColumnTransformer)
    gb_model = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
    )

    gb_pipe = Pipeline([
        ("preprocess", preprocessor),
        ("model", gb_model),
    ])

    if sample_weight is not None:
        gb_pipe.fit(X_train, y_train, model__sample_weight=sample_weight)
    else:
        gb_pipe.fit(X_train, y_train)

    gb_preds = gb_pipe.predict(X_test)
    gb_proba = gb_pipe.predict_proba(X_test)[:, 1]

    gb_acc = accuracy_score(y_test, gb_preds)
    gb_roc = roc_auc_score(y_test, gb_proba)

    report_gb = classification_report(y_test, gb_preds, output_dict=True)
    if verbose:
        print("\nGradientBoostingClassifier" + (" (class_weight=balanced)" if use_class_weight else ""))
        print(classification_report(y_test, gb_preds))
        print("Confusion matrix:\n", confusion_matrix(y_test, gb_preds))

    gb_f1_macro = report_gb["macro avg"]["f1-score"]
    gb_f1_weighted = report_gb["weighted avg"]["f1-score"]
    gb_f1_pos = report_gb["1"]["f1-score"] if "1" in report_gb else None

    best_thr_gb = None
    best_thr_metrics_gb: Dict[str, float] = {}
    if tune_threshold:
        best_thr_metrics_gb = _find_best_threshold(y_test.values, gb_proba)
        best_thr_gb = best_thr_metrics_gb["threshold"]
        _plot_threshold_curve(
            y_test.values,
            gb_proba,
            model_name="GradientBoostingClassifier Binary Satisfaction",
        )

    results.append({
        "model": "GradientBoostingClassifier",
        "accuracy": gb_acc,
        "roc_auc": gb_roc,
        "f1_macro": gb_f1_macro,
        "f1_weighted": gb_f1_weighted,
        "f1_positive": gb_f1_pos,
        "best_threshold": best_thr_gb,
        "f1_at_best_threshold": best_thr_metrics_gb.get("f1") if best_thr_metrics_gb else None,
        "precision_at_best_threshold": best_thr_metrics_gb.get("precision") if best_thr_metrics_gb else None,
        "recall_at_best_threshold": best_thr_metrics_gb.get("recall") if best_thr_metrics_gb else None,
    })

    # 2) CatBoost trained directly on preprocessed arrays (avoid sklearn Pipeline issues)
    if HAS_CATBOOST:
        X_train_enc = preprocessor.fit_transform(X_train)
        X_test_enc = preprocessor.transform(X_test)

        cat_model = CatBoostClassifier(
            iterations=400,
            depth=6,
            learning_rate=0.05,
            loss_function="Logloss",
            verbose=False,
            random_state=42,
            auto_class_weights="Balanced",
        )

        # CatBoost also accepts sample weights for class balancing
        if sample_weight is not None:
            cat_model.fit(X_train_enc, y_train, sample_weight=sample_weight)
        else:
            cat_model.fit(X_train_enc, y_train)

        cat_preds = cat_model.predict(X_test_enc).ravel().astype(int)
        cat_proba = cat_model.predict_proba(X_test_enc)[:, 1]

        cat_acc = accuracy_score(y_test, cat_preds)
        cat_roc = roc_auc_score(y_test, cat_proba)

        report_cat = classification_report(y_test, cat_preds, output_dict=True)

        if verbose:
            print("\nCatBoostClassifier" + (" (auto_class_weights=Balanced)" if use_class_weight else ""))
            print(classification_report(y_test, cat_preds))
            print("Confusion matrix:\n", confusion_matrix(y_test, cat_preds))

        cat_f1_macro = report_cat["macro avg"]["f1-score"]
        cat_f1_weighted = report_cat["weighted avg"]["f1-score"]
        cat_f1_pos = report_cat["1"]["f1-score"] if "1" in report_cat else None

        best_thr_cat = None
        best_thr_metrics_cat: Dict[str, float] = {}
        if tune_threshold:
            best_thr_metrics_cat = _find_best_threshold(y_test.values, cat_proba)
            best_thr_cat = best_thr_metrics_cat["threshold"]
            _plot_threshold_curve(
                y_test.values,
                cat_proba,
                model_name="CatBoostClassifier Binary Satisfaction",
            )

        results.append({
            "model": "CatBoostClassifier",
            "accuracy": cat_acc,
            "roc_auc": cat_roc,
            "f1_macro": cat_f1_macro,
            "f1_weighted": cat_f1_weighted,
            "f1_positive": cat_f1_pos,
            "best_threshold": best_thr_cat,
            "f1_at_best_threshold": best_thr_metrics_cat.get("f1") if best_thr_metrics_cat else None,
            "precision_at_best_threshold": best_thr_metrics_cat.get("precision") if best_thr_metrics_cat else None,
            "recall_at_best_threshold": best_thr_metrics_cat.get("recall") if best_thr_metrics_cat else None,
        })

    return pd.DataFrame(results)


def build_and_train_clv_model(clv_df: pd.DataFrame) -> Dict[str, float]:
    """Optional: train a regression model on top 5% customers by revenue."""

    from sklearn.ensemble import GradientBoostingRegressor

    df = clv_df.dropna(subset=["customer_total_revenue"]).copy()
    threshold = df["customer_total_revenue"].quantile(0.95)
    top = df[df["customer_total_revenue"] >= threshold]

    X = top[["customer_n_orders", "customer_avg_order_value"]]
    y = top["customer_total_revenue"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    model = GradientBoostingRegressor(random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure()
    sns.scatterplot(x=y_test, y=y_pred)
    plt.xlabel("Actual CLV")
    plt.ylabel("Predicted CLV")
    plt.title("CLV Prediction: Actual vs Predicted (Top 5% Customers)")
    save_figure("clv_actual_vs_predicted")
    plt.show()

    return {"rmse": rmse, "mae": mae}
