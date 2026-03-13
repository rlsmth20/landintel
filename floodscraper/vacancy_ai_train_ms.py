from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from vacancy_ai_common import MODEL_METRICS_PATH, MODEL_PATH, MODEL_VERSION, TRAINING_MANIFEST_PATH, feature_columns, write_metrics


def train_model(*, manifest_path: str, output_model: str, output_metrics: str) -> None:
    frame = pd.read_parquet(manifest_path)
    columns = feature_columns(frame)
    x = frame.loc[:, columns]
    y = pd.to_numeric(frame["weak_building_label"], errors="coerce").astype(int)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    pipeline.fit(x_train, y_train)

    probability = pipeline.predict_proba(x_test)[:, 1]
    predicted = (probability >= 0.5).astype(int)
    metrics = {
        "model_version": MODEL_VERSION,
        "manifest_path": manifest_path,
        "feature_count": len(columns),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "accuracy": float(round(accuracy_score(y_test, predicted), 4)),
        "precision": float(round(precision_score(y_test, predicted, zero_division=0), 4)),
        "recall": float(round(recall_score(y_test, predicted, zero_division=0), 4)),
        "roc_auc": float(round(roc_auc_score(y_test, probability), 4)),
    }
    joblib.dump({"pipeline": pipeline, "feature_columns": columns, "model_version": MODEL_VERSION}, output_model)
    write_metrics(Path(output_metrics), metrics)
    print(f"Trained {MODEL_VERSION} with {len(columns)} features; metrics written to {output_metrics}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Mississippi building-presence classifier from weakly labeled imagery.")
    parser.add_argument("--manifest", default=str(TRAINING_MANIFEST_PATH))
    parser.add_argument("--output-model", default=str(MODEL_PATH))
    parser.add_argument("--output-metrics", default=str(MODEL_METRICS_PATH))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(manifest_path=args.manifest, output_model=args.output_model, output_metrics=args.output_metrics)
