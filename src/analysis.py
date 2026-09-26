from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


class KNNResultsAnalyzer:
    """Create aggregate tables and visualizations from KNN evaluations."""

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self.df = pd.read_csv(self.file_path, index_col=0)

    def compute_statistics(self) -> pd.DataFrame:
        stats = self.df.describe().transpose()[["mean", "std"]]
        output = Path("data/output/knn_results_statistics.csv")
        output.parent.mkdir(parents=True, exist_ok=True)
        stats.to_csv(output)
        return stats

    def create_summary_table(self) -> pd.DataFrame:
        summary = pd.DataFrame(
            {
                "Distance Metric": ["Euclidean", "Cosine"],
                "Accuracy": [
                    self.df["euclidean_top1"].mean(),
                    self.df["cosine_top1"].mean(),
                ],
                "F1-Macro": [
                    self.df["euclidean_f1_macro"].mean(),
                    self.df["cosine_f1_macro"].mean(),
                ],
                "F1-Weighted": [
                    self.df["euclidean_f1_weighted"].mean(),
                    self.df["cosine_f1_weighted"].mean(),
                ],
                "AUC": [
                    self.df["euclidean_auc"].mean(),
                    self.df["cosine_auc"].mean(),
                ],
            }
        )
        output = Path("data/output/knn_summary_table.csv")
        output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output, index=False)
        return summary

    def plot_auc_comparison(self) -> Path:
        output = Path("data/output/plots/auc_comparison.png")
        output.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.plot(self.df.index, self.df["euclidean_auc"], marker="o", label="Euclidean")
        plt.plot(self.df.index, self.df["cosine_auc"], marker="o", label="Cosine")
        plt.xlabel("Number of Neighbors (k)")
        plt.ylabel("Weighted ROC-AUC")
        plt.title("KNN ROC-AUC by Distance Metric")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(output, dpi=300)
        plt.close()
        return output

    def analyze(self):
        return {
            "statistics": self.compute_statistics(),
            "summary": self.create_summary_table(),
            "auc_plot": self.plot_auc_comparison(),
        }
