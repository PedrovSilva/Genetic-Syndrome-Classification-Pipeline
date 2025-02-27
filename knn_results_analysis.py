import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class KNNResultsAnalyzer:
    def __init__(self, file_path):
        """Initialize the class by reading the CSV file."""
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
    
    def compute_statistics(self):
        """Compute mean and standard deviation for each metric."""
        stats = self.df.describe().transpose()[['mean', 'std']]
        stats.to_csv("data/output/knn_results_statistics.csv")
        print("Statistics saved as knn_results_statistics.csv")
        return stats
    
    def create_summary_table(self):
        """Create a summary table for Euclidean and Cosine metrics."""
        summary_data = {
            "Distance Metric": ["Euclidean", "Cosine"],
            "Accuracy": [self.df["euclidean_top1"].mean(), self.df["cosine_top1"].mean()],
            "F1-Score": [self.df["euclidean_f1"].mean(), self.df["cosine_f1"].mean()],
            "AUC": [self.df["euclidean_auc"].mean(), self.df["cosine_auc"].mean()]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_path = "data/output/knn_summary_table.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary table saved as {summary_path}")
        return summary_df
    
    def plot_metrics_comparison(self):
        """Plot comparison of metrics for Cosine and Euclidean distances."""
        plt.figure(figsize=(10, 6))
        for metric in ['euclidean_auc', 'cosine_auc']:
            plt.plot(self.df.index, self.df[metric], marker='o', label=metric)
        
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('AUC Score')
        plt.title('Comparison of AUC for Euclidean and Cosine Distance')
        plt.legend()
        plt.grid()
        output_dir = "data/output/plots"  
        os.makedirs(output_dir, exist_ok=True)  
        plt.savefig(os.path.join(output_dir, "auc_comparison.png")) 
        print("AUC comparison plot saved as auc_comparison.png")
    
    def analyze(self):
        """Run all analyses and save outputs."""
        stats = self.compute_statistics()
        summary = self.create_summary_table()
        self.plot_metrics_comparison()
        return stats, summary
