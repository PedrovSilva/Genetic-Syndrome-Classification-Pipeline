import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, f1_score, top_k_accuracy_score
from tqdm import tqdm

class KNNClassifier:
    def __init__(self, X, y, k_range=range(1, 16), cv_folds=10):
        """
        Initializes the KNN classifier with embeddings and labels.

        Parameters:
        - X: np.array -> Image embeddings.
        - y: np.array -> Labels (syndrome_id).
        - k_range: range -> Range of K values for KNN.
        - cv_folds: int -> Number of folds for cross-validation.
        """
        self.X = X
        self.y = y
        self.k_range = k_range
        self.cv_folds = cv_folds
        self.results = {}

    def evaluate_knn(self, metric="euclidean"):
        """
        Evaluates KNN for different values of K using cross-validation.

        Parameters:
        - metric: str -> "euclidean" or "cosine", defines the distance metric.

        Returns:
        - DataFrame with the average results for each K value.
        """
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        metrics_results = {k: {"auc": [], "f1": [], "top1": [], "top3": []} for k in self.k_range}

        print(f"\n🔍 Evaluating KNN with metric: {metric}")
        for k in tqdm(self.k_range, desc=f"Evaluating KNN ({metric})"):
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
            
            fold_auc, fold_f1, fold_top1, fold_top3 = [], [], [], []
            
            for train_idx, test_idx in skf.split(self.X, self.y):
                X_train, X_test = self.X[train_idx], self.X[test_idx]
                y_train, y_test = self.y[train_idx], self.y[test_idx]

                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                y_proba = knn.predict_proba(X_test)

                # Compute metrics
                auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
                f1 = f1_score(y_test, y_pred, average="weighted")
                top1 = top_k_accuracy_score(y_test, y_proba, k=1)
                top3 = top_k_accuracy_score(y_test, y_proba, k=3)

                fold_auc.append(auc)
                fold_f1.append(f1)
                fold_top1.append(top1)
                fold_top3.append(top3)

            # Compute the average metrics for each K value
            metrics_results[k]["auc"] = np.mean(fold_auc)
            metrics_results[k]["f1"] = np.mean(fold_f1)
            metrics_results[k]["top1"] = np.mean(fold_top1)
            metrics_results[k]["top3"] = np.mean(fold_top3)

        self.results[metric] = pd.DataFrame(metrics_results).T
        return self.results[metric]

    def compare_metrics(self):
        """
        Compares the results between Euclidean and Cosine distance metrics.
        Returns a consolidated DataFrame.
        """
        if "euclidean" in self.results and "cosine" in self.results:
            df_euc = self.results["euclidean"].add_prefix("euclidean_")
            df_cos = self.results["cosine"].add_prefix("cosine_")
            df_combined = pd.concat([df_euc, df_cos], axis=1)
            return df_combined
        else:
            print("Please run evaluation with both distance metrics first!")
            return None
