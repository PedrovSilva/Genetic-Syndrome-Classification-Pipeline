from dataset import GeneticSyndromeDataset
import numpy as np
from dataset import GeneticSyndromeDataset
from knn_classifier import KNNClassifier
from knn_results_analysis import KNNResultsAnalyzer


def analysis():
    analyzer = KNNResultsAnalyzer("results_knn_comparison.csv")
    stats = analyzer.analyze()
def classification():
    # Load dataset
    file_path = "data/mini_gm_public_v0.1.p"
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
    df_euclidean.to_csv("results_knn_euclidean.csv", index=True)
    df_cosine.to_csv("results_knn_cosine.csv", index=True)

    # Save comparison table
    df_comparison.to_csv("results_knn_comparison.csv", index=True)

    print("\n📁 Results saved:")
    print("- results_knn_euclidean.csv")
    print("- results_knn_cosine.csv")
    print("- results_knn_comparison.csv")


def data_visualization():
    file_path = "data/mini_gm_public_v0.1.p"

    print("Iniciando o carregamento dos dados...")
    dataset = GeneticSyndromeDataset(file_path)

    print("\nExibindo resumo dos dados:")
    dataset.get_summary()

    print("\nGerando visualização dos embeddings com t-SNE...")
    dataset.plot_tsne()

if __name__ == "__main__":
    #data_visualization()
    #classification()
    analysis()
