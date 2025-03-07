import os
from dataset import GeneticSyndromeDataset
import numpy as np
from dataset import GeneticSyndromeDataset
from knn_classifier import KNNClassifier
from knn_results_analysis import KNNResultsAnalyzer


def analysis():
    analyzer = KNNResultsAnalyzer("data/output/results_knn_comparison.csv")
    stats = analyzer.analyze()

def classification(file_path = "data/input/mini_gm_public_v0.1.p"):
    # Load dataset
    
    dataset = GeneticSyndromeDataset(file_path)

    # Convert embeddings to NumPy arrays
    X = np.vstack(dataset.df["embedding"].values)
    y = dataset.df["syndrome_id"].values

    # Initialize classifier
    knn_classifier = KNNClassifier(X, y)

    # Evaluate with Euclidean distance
    df_euclidean = knn_classifier.evaluate_knn(metric="euclidean")

    # Evaluate with Cosine distance
    df_cosine = knn_classifier.evaluate_knn(metric="cosine")

    # Compare results
    df_comparison = knn_classifier.compare_metrics()
    print("\n🔍 Distance Metric Comparison:")
    print(df_comparison)

    # Save individual metric results
    output_dir = "data/output"
    os.makedirs(output_dir, exist_ok=True)

    df_euclidean.to_csv(os.path.join(output_dir, "results_knn_euclidean.csv"), index=True)
    df_cosine.to_csv(os.path.join(output_dir, "results_knn_cosine.csv"), index=True)

    # Save comparison table
    df_comparison.to_csv(os.path.join(output_dir, "results_knn_comparison.csv"), index=True)

    print("\n📁 Results saved on: data/output/")
    print("- results_knn_euclidean.csv")
    print("- results_knn_cosine.csv")
    print("- results_knn_comparison.csv")


def data_visualization(file_path = "data/input/mini_gm_public_v0.1.p"):
    

    print("Starting data loading...")
    dataset = GeneticSyndromeDataset(file_path)

    print("\nDisplaying data summary:")
    dataset.get_summary()

    print("\nGenerating embedding visualization with t-SNE...")
    dataset.plot_tsne()

if __name__ == "__main__":

    data_visualization()
    classification()
    analysis()
