from __future__ import annotations

import io
import math
import os
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

app = FastAPI(title="Student Performance Analyzer API", version="1.0.0")
frontend_origins = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000,https://student-performance-analyzer-ui.vercel.app/").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUBJECT_COLUMNS = [
    "Math_Stats", "Machine_Learning", "Python_Prog",
    "Database_Mgmt", "Data_Structures"
]
ID_COLUMNS = {"Student_ID", "Student_Name", "Name", "ID"}

def json_safe(value: Any):
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value

def clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # Empty strings should be treated as missing values.
    df = df.replace(r"^\s*$", np.nan, regex=True)
    # Drop completely empty rows/columns.
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return df

def prepare_target(df: pd.DataFrame, target: str):
    if target not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target}' was not found.")
    y = pd.to_numeric(df[target], errors="coerce")
    valid = y.notna()
    df, y = df.loc[valid].copy(), y.loc[valid].astype(float)
    if len(df) < 12:
        raise HTTPException(status_code=400, detail="At least 12 rows with a numeric target are required.")
    if y.nunique() < 2:
        raise HTTPException(status_code=400, detail="The target must contain at least two distinct numeric values.")
    return df, y

def build_pipeline(X: pd.DataFrame, estimator):
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipe, numeric))
    if categorical:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", categorical_pipe, categorical))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])

def get_feature_names(pipeline):
    try:
        names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        return [str(n).replace("numeric__", "").replace("categorical__", "") for n in names]
    except Exception:
        return []

def analyze_dataframe(raw_df: pd.DataFrame, target: str = "Average_Score"):
    df = clean_frame(raw_df)
    if df.empty or len(df.columns) < 2:
        raise HTTPException(status_code=400, detail="CSV needs at least two non-empty columns.")
    if target not in df.columns:
        # Helpful default: first numeric column with a score-like name, otherwise last numeric column.
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        candidate = next((c for c in numeric_cols if "average" in c.lower() or "score" in c.lower()), None)
        if candidate:
            target = candidate
        else:
            raise HTTPException(status_code=400, detail=f"Target '{target}' not found. Available columns: {list(df.columns)}")

    df, y = prepare_target(df, target)
    df = df.reset_index(drop=True)
    X = df.drop(columns=[target])

    # Avoid identity leakage and known target leakage for this sample schema.
    excluded = [c for c in X.columns if c in ID_COLUMNS]
    if target == "Average_Score":
        excluded += [c for c in SUBJECT_COLUMNS if c in X.columns]
    X = X.drop(columns=list(set(excluded)), errors="ignore")
    X = X.drop(columns=[c for c in X.columns if X[c].nunique(dropna=True) <= 1], errors="ignore")
    if X.shape[1] == 0:
        raise HTTPException(status_code=400, detail="No usable predictor columns remain after removing IDs and leakage-prone fields.")

    # Keep original row indices for the prediction table.
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    if len(X_train) < 8:
        raise HTTPException(status_code=400, detail="Not enough rows for a train/test split.")

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, max_depth=None, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=2,
            loss="squared_error", random_state=42
        ),
    }

    model_results = []
    fitted = {}
    cv_splits = min(5, max(2, len(X_train) // 5))
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

    for name, estimator in models.items():
        pipe = build_pipeline(X_train, estimator)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        try:
            cv_r2 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2", n_jobs=None)
            cv_mean = float(np.mean(cv_r2))
            cv_std = float(np.std(cv_r2))
        except Exception:
            cv_mean, cv_std = None, None
        model_results.append({
            "name": name,
            "r2": json_safe(r2_score(y_test, preds)),
            "mae": json_safe(mean_absolute_error(y_test, preds)),
            "rmse": json_safe(np.sqrt(mean_squared_error(y_test, preds))),
            "cv_r2_mean": json_safe(cv_mean),
            "cv_r2_std": json_safe(cv_std),
        })
        fitted[name] = (pipe, preds)

    # Highest held-out R² is highlighted, while metrics remain visible for context.
    best_name = max(model_results, key=lambda m: m["r2"] if m["r2"] is not None else -1)["name"]
    best_pipe, best_preds = fitted[best_name]

    # Aggregate feature importance from tree model when available; fallback to permutation-like
    # drop-in-score is intentionally avoided to keep this endpoint fast and deterministic.
    tree_pipe = fitted["Random Forest"][0]
    try:
        names = get_feature_names(tree_pipe)
        importances = tree_pipe.named_steps["model"].feature_importances_.tolist()
        feature_importance = sorted(
            [{"feature": n, "importance": json_safe(v)} for n, v in zip(names, importances)],
            key=lambda x: x["importance"] or 0, reverse=True
        )[:12]
    except Exception:
        feature_importance = []

    numeric_df = df.select_dtypes(include="number")
    correlations = []
    if target in numeric_df.columns:
        corr = numeric_df.corr(numeric_only=True)[target].drop(labels=[target], errors="ignore").dropna()
        correlations = [
            {"feature": str(k), "correlation": json_safe(v)}
            for k, v in corr.reindex(corr.abs().sort_values(ascending=False).index).head(10).items()
        ]

    # Pairwise views include numeric scatter data and category-level target means.
    # Keep chart payloads bounded and omit identifier-like/high-cardinality categories.
    relationships = []
    for col in numeric_df.columns:
        if col == target or col in ID_COLUMNS:
            continue
        pair = pd.DataFrame({"x": pd.to_numeric(df[col], errors="coerce"), "y": y}).dropna()
        if len(pair) < 2:
            continue
        correlation = pair["x"].corr(pair["y"])
        if len(pair) > 160:
            sample_indices = np.linspace(0, len(pair) - 1, 160, dtype=int)
            pair = pair.iloc[sample_indices]
        relationships.append({
            "feature": str(col),
            "kind": "numeric",
            "correlation": json_safe(correlation),
            "points": [{"x": json_safe(row.x), "y": json_safe(row.y)} for row in pair.itertuples(index=False)],
            "groups": [],
        })

    for col in df.select_dtypes(exclude="number").columns:
        if col in ID_COLUMNS or df[col].nunique(dropna=True) > 12:
            continue
        pair = pd.DataFrame({"group": df[col].astype("string"), "y": y}).dropna()
        if pair["group"].nunique() < 2:
            continue
        grouped = pair.groupby("group", observed=True)["y"].agg(["mean", "count"]).sort_values("mean")
        relationships.append({
            "feature": str(col),
            "kind": "category",
            "correlation": None,
            "points": [],
            "groups": [
                {"label": str(label), "average": json_safe(row["mean"]), "count": int(row["count"])}
                for label, row in grouped.iterrows()
            ],
        })

    # Subject-level summary is descriptive only, not used to predict Average_Score.
    subject_summary = []
    for col in SUBJECT_COLUMNS:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals):
                subject_summary.append({
                    "subject": col.replace("_", " "),
                    "average": json_safe(vals.mean()),
                    "min": json_safe(vals.min()),
                    "max": json_safe(vals.max()),
                })

    score_bins = pd.cut(y, bins=min(8, max(3, y.nunique())), duplicates="drop")
    distribution = []
    for interval, count in score_bins.value_counts(sort=False).items():
        distribution.append({"range": str(interval), "count": int(count)})

    prediction_rows = []
    for i, (actual, predicted) in enumerate(zip(y_test.tolist(), best_preds.tolist())):
        prediction_rows.append({
            "row": int(i + 1),
            "actual": json_safe(actual),
            "predicted": json_safe(predicted),
            "residual": json_safe(actual - predicted),
        })

    # Row-level summaries support human-readable, descriptive student insights.
    # These are screening cues only, not validated diagnoses or causal explanations.
    available_subjects = [col for col in SUBJECT_COLUMNS if col in df.columns]
    if target == "Average_Score":
        bands = [60, 75, 85]
    else:
        bands = [float(y.quantile(q)) for q in (0.25, 0.5, 0.75)]
    segment_names = ["Needs support", "Developing", "On track", "High achieving"]

    student_insights = []
    for row_index, (_, row) in enumerate(df.iterrows()):
        score = float(y.iloc[row_index])
        segment = segment_names[sum(score >= boundary for boundary in bands)]
        subjects = {}
        for col in available_subjects:
            subject_score = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            if pd.notna(subject_score):
                subjects[col.replace("_", " ")] = float(subject_score)
        weakest = min(subjects, key=subjects.get) if subjects else None
        focus_reasons = []
        attendance = pd.to_numeric(pd.Series([row.get("Attendance_%")]), errors="coerce").iloc[0]
        study_hours = pd.to_numeric(pd.Series([row.get("Weekly_Study_Hours")]), errors="coerce").iloc[0]
        if weakest and subjects[weakest] < 70:
            focus_reasons.append(f"review core concepts in {weakest}")
        if pd.notna(attendance) and attendance < 75:
            focus_reasons.append("check attendance barriers and agree a support plan")
        if not focus_reasons and weakest:
            focus_reasons.append(f"build confidence in {weakest} with short, regular practice")
        if not focus_reasons:
            focus_reasons.append("keep the current study routine and check progress next assessment")

        raw_name = row.get("Student_Name", row.get("Name"))
        raw_id = row.get("Student_ID", row.get("ID"))
        student_label = str(raw_name) if pd.notna(raw_name) else (str(raw_id) if pd.notna(raw_id) else f"Student {row_index + 1}")
        identifier = str(raw_id) if pd.notna(raw_id) else f"Record {row_index + 1}"
        insight = f"{student_label} is currently {segment.lower()} with a {target} of {score:.1f}. Suggested next step: {focus_reasons[0]}."
        if pd.notna(study_hours) and study_hours < 10:
            insight += " Consider setting a manageable weekly study target together."

        student_insights.append({
            "id": identifier,
            "name": student_label,
            "score": json_safe(score),
            "segment": segment,
            "focus_area": weakest or "General progress",
            "focus_score": json_safe(subjects.get(weakest)) if weakest else None,
            "attendance": json_safe(attendance),
            "study_hours": json_safe(study_hours),
            "insight": insight,
        })

    student_insights.sort(key=lambda student: student["score"] if student["score"] is not None else float("inf"))

    columns = []
    for col in df.columns:
        sample_values = [json_safe(v) for v in df[col].dropna().head(3).tolist()]
        columns.append({
            "name": str(col),
            "dtype": str(df[col].dtype),
            "missing": int(df[col].isna().sum()),
            "unique": int(df[col].nunique(dropna=True)),
            "sample": sample_values,
        })

    return {
        "dataset": {
            "rows": int(len(df)),
            "columns_count": int(len(df.columns)),
            "predictor_count": int(X.shape[1]),
            "target": target,
            "target_mean": json_safe(y.mean()),
            "target_median": json_safe(y.median()),
            "target_std": json_safe(y.std()),
            "missing_cells": int(df.isna().sum().sum()),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "excluded_columns": sorted(set(excluded)),
            "predictors": X.columns.tolist(),
            "columns": columns,
        },
        "models": model_results,
        "best_model": best_name,
        "feature_importance": feature_importance,
        "correlations": correlations,
        "relationships": relationships,
        "subject_summary": subject_summary,
        "distribution": distribution,
        "predictions": prediction_rows,
        "student_insights": student_insights,
    }

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "Student Performance Analyzer"}

@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), target: str = "Average_Score"):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")
    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        df = pd.read_csv(io.BytesIO(content))
        return analyze_dataframe(df, target=target)
    except HTTPException:
        raise
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV has no readable data.")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Could not decode CSV. Save it as UTF-8 and retry.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Analysis failed: {str(exc)}")
