from pathlib import Path

from src.analysis import KNNResultsAnalyzer
from src.dataset import GeneticSyndromeDataset
from src.models.knn_classifier import KNNClassifier


DATASET_PATH = Path("data/input/mini_gm_public_v0.1.p")


def main() -> None:
    dataset = GeneticSyndromeDataset(DATASET_PATH)

    print("Dataset summary:")
    for key, value in dataset.get_summary().items():
        print(f"  {key}: {value}")

    print("\nGenerating t-SNE visualization...")
    dataset.plot_tsne()

    classifier = KNNClassifier(
        X=dataset.X,
        y=dataset.y,
        subject_ids=dataset.subject_ids,
    )

    classifier.evaluate_knn("euclidean")
    classifier.evaluate_knn("cosine")
    classifier.save_results()

    analyzer = KNNResultsAnalyzer("data/output/results_knn_comparison.csv")
    analyzer.analyze()

    print("\nPipeline completed. Results are available under data/output/.")


if __name__ == "__main__":
    main()
