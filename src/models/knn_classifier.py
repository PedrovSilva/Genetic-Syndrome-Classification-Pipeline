from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm


class KNNClassifier:
    """Evaluate KNN while keeping all images from a subject in one fold."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subject_ids: np.ndarray,
        k_range=range(1, 16),
        cv_folds: int = 5,
        random_state: int = 42,
    ):
        self.X = X
        self.y = y
        self.subject_ids = subject_ids
        self.k_range = k_range
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.results: dict[str, pd.DataFrame] = {}

    def evaluate_knn(self, metric: str = "euclidean") -> pd.DataFrame:
        if metric not in {"euclidean", "cosine"}:
            raise ValueError("metric must be 'euclidean' or 'cosine'")

        splitter = StratifiedGroupKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )
        rows = []

        for k in tqdm(self.k_range, desc=f"Evaluating KNN ({metric})"):
            fold_metrics = []

            for train_idx, test_idx in splitter.split(
                self.X, self.y, groups=self.subject_ids
            ):
                model = KNeighborsClassifier(n_neighbors=k, metric=metric)
                model.fit(self.X[train_idx], self.y[train_idx])

                y_true = self.y[test_idx]
                y_pred = model.predict(self.X[test_idx])
                y_proba = model.predict_proba(self.X[test_idx])

                fold_metrics.append(
                    {
                        "auc": roc_auc_score(
                            y_true,
                            y_proba,
                            multi_class="ovr",
                            average="weighted",
                        ),
                        "f1_macro": f1_score(
                            y_true, y_pred, average="macro", zero_division=0
                        ),
                        "f1_weighted": f1_score(
                            y_true, y_pred, average="weighted", zero_division=0
                        ),
                        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
                        "top1": top_k_accuracy_score(y_true, y_proba, k=1),
                        "top3": top_k_accuracy_score(y_true, y_proba, k=3),
                    }
                )

            mean_metrics = pd.DataFrame(fold_metrics).mean()
            rows.append({"k": k, **mean_metrics.to_dict()})

        result = pd.DataFrame(rows).set_index("k")
        self.results[metric] = result
        return result

    def compare_metrics(self) -> pd.DataFrame:
        if set(self.results) != {"euclidean", "cosine"}:
            raise RuntimeError("Evaluate both euclidean and cosine metrics first.")

        euclidean = self.results["euclidean"].add_prefix("euclidean_")
        cosine = self.results["cosine"].add_prefix("cosine_")
        return pd.concat([euclidean, cosine], axis=1)

    def save_results(self, output_dir: str | Path = "data/output") -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for metric, result in self.results.items():
            result.to_csv(output_path / f"results_knn_{metric}.csv")

        comparison = self.compare_metrics()
        comparison.to_csv(output_path / "results_knn_comparison.csv")

        return output_path / "results_knn_comparison.csv"
