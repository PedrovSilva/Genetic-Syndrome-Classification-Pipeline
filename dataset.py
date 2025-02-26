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
        Inicializa a classe carregando os dados do arquivo pickle.
        """
        self.file_path = file_path
        self.data = self._load_data()
        self.df = self._flatten_data()

    def _load_data(self):
        """ Carrega os dados do arquivo pickle. """
        try:
            with open(self.file_path, "rb") as f:
                data = pickle.load(f)
            print(f"Dados carregados com sucesso! Total de síndromes: {len(data)}")
            return data
        except Exception as e:
            print(f"Erro ao carregar os dados: {e}")
            return None

    def _flatten_data(self):
        """ Converte a estrutura hierárquica em um DataFrame. """
        if self.data is None:
            return None

        rows = []
        for syndrome_id, subjects in self.data.items():
            for subject_id, images in subjects.items():
                for image_id, embedding in images.items():
                    rows.append([syndrome_id, subject_id, image_id, np.array(embedding)])

        df = pd.DataFrame(rows, columns=["syndrome_id", "subject_id", "image_id", "embedding"])
        print(f"Dataset formatado com {len(df)} entradas.")
        return df

    def get_summary(self):
        """ Exibe estatísticas básicas do dataset. """
        if self.df is None:
            print("Os dados não foram carregados corretamente.")
            return

        print("\nResumo dos dados:")
        print(self.df.head())
        print("\nDistribuição de síndromes:")
        print(self.df["syndrome_id"].value_counts())
        print("\nValores nulos:")
        print(self.df.isnull().sum())

    def plot_tsne(self, perplexity=30, learning_rate=200, random_state=42, save_path="data/output/plots/tsne_visualization.png"):
        """ Reduz a dimensionalidade dos embeddings para 2D usando t-SNE e salva o gráfico. """
        if self.df is None:
            print("Os dados não foram carregados corretamente.")
            return

        print("Executando t-SNE para redução de dimensionalidade...")

        # Criar diretório se não existir
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Extrair embeddings e síndrome associada
        X = np.vstack(self.df["embedding"].values)
        y = self.df["syndrome_id"].values

        # Aplicar t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=random_state)
        X_embedded = tsne.fit_transform(X)

        # Criar DataFrame com resultados
        tsne_df = pd.DataFrame({"tsne_1": X_embedded[:, 0], "tsne_2": X_embedded[:, 1], "syndrome_id": y})

        # Criar a figura
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x="tsne_1", y="tsne_2",
            hue="syndrome_id",
            palette=sns.color_palette("hsv", len(set(y))),
            data=tsne_df,
            alpha=0.7
        )
        plt.title("Visualização de Embeddings usando t-SNE")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Syndrome ID")

        # Salvar figura
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Gráfico salvo em: {save_path}")

        # Mostrar gráfico opcionalmente
        plt.show()
