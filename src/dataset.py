from __future__ import annotations

import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


class GeneticSyndromeDataset:
    """Load and flatten the hierarchical genetic syndrome embedding dataset."""

    COLUMNS = ["syndrome_id", "subject_id", "image_id", "embedding"]

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)
        self.data = self._load_data()
        self.df = self._flatten_data()

    def _load_data(self) -> dict:
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.file_path}")

        with self.file_path.open("rb") as file:
            data = pickle.load(file)

        if not isinstance(data, dict):
            raise ValueError("Expected the pickle dataset to contain a dictionary.")

        return data

    def _flatten_data(self) -> pd.DataFrame:
        rows = []
        for syndrome_id, subjects in self.data.items():
            for subject_id, images in subjects.items():
                for image_id, embedding in images.items():
                    rows.append(
                        [syndrome_id, subject_id, image_id, np.asarray(embedding)]
                    )

        return pd.DataFrame(rows, columns=self.COLUMNS)

    @property
    def X(self) -> np.ndarray:
        return np.vstack(self.df["embedding"].values)

    @property
    def y(self) -> np.ndarray:
        return self.df["syndrome_id"].to_numpy()

    @property
    def subject_ids(self) -> np.ndarray:
        return self.df["subject_id"].to_numpy()

    def get_summary(self) -> dict:
        return {
            "samples": len(self.df),
            "syndromes": self.df["syndrome_id"].nunique(),
            "subjects": self.df["subject_id"].nunique(),
            "missing_values": int(self.df.isna().sum().sum()),
            "class_distribution": self.df["syndrome_id"].value_counts().to_dict(),
        }

    def plot_tsne(
        self,
        perplexity: float = 30,
        learning_rate: float = 200,
        random_state: int = 42,
        save_path: str | Path = "data/output/plots/tsne_visualization.png",
    ) -> Path:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        embedded = tsne.fit_transform(self.X)

        plt.figure(figsize=(10, 6))
        for syndrome_id in np.unique(self.y):
            mask = self.y == syndrome_id
            plt.scatter(
                embedded[mask, 0],
                embedded[mask, 1],
                alpha=0.7,
                label=str(syndrome_id),
            )

        plt.title("Embedding Visualization using t-SNE")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Syndrome ID")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path
