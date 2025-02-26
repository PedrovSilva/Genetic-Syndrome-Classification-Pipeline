import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

class GeneticSyndromeDataset:
    def __init__(self, file_path):
        """
        Initializes the class by loading the data from the pickle file.
        """
        self.file_path = file_path
        self.data = self._load_data()
        self.df = self._flatten_data()

    def _load_data(self):
        """ Loads the data from the pickle file. """
        try:
            with open(self.file_path, "rb") as f:
                data = pickle.load(f)
            print(f"Data loaded successfully! Total number of syndromes: {len(data)}")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def _flatten_data(self):
        """ Converts the hierarchical structure into a DataFrame. """
        if self.data is None:
            return None

        rows = []
        for syndrome_id, subjects in self.data.items():
            for subject_id, images in subjects.items():
                for image_id, embedding in images.items():
                    rows.append([syndrome_id, subject_id, image_id, np.array(embedding)])

        df = pd.DataFrame(rows, columns=["syndrome_id", "subject_id", "image_id", "embedding"])
        print(f"Dataset formatted with {len(df)} entries.")
        return df

    def get_summary(self):
        """ Displays basic statistics of the dataset. """
        if self.df is None:
            print("The data was not loaded correctly.")
            return

        print("\nData Summary:")
        print(self.df.head())
        print("\nSyndrome distribution:")
        print(self.df["syndrome_id"].value_counts())
        print("\nNull values:")
        print(self.df.isnull().sum())

    def plot_tsne(self, perplexity=30, learning_rate=200, random_state=42, save_path="data/output/plots/tsne_visualization.png"):
        """ Reduces the dimensionality of the embeddings to 2D using t-SNE and saves the plot. """
        if self.df is None:
            print("The data was not loaded correctly.")
            return

        print("Running t-SNE for dimensionality reduction...")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Extract embeddings and associated syndrome
        X = np.vstack(self.df["embedding"].values)
        y = self.df["syndrome_id"].values

        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=random_state)
        X_embedded = tsne.fit_transform(X)

        # Create DataFrame with results
        tsne_df = pd.DataFrame({"tsne_1": X_embedded[:, 0], "tsne_2": X_embedded[:, 1], "syndrome_id": y})

        # Create the figure
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x="tsne_1", y="tsne_2",
            hue="syndrome_id",
            palette=sns.color_palette("hsv", len(set(y))),
            data=tsne_df,
            alpha=0.7
        )
        plt.title("Embedding Visualization using t-SNE")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Syndrome ID")

        # Save the figure
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Plot saved at: {save_path}")

        # Optionally show the plot
        plt.show()
