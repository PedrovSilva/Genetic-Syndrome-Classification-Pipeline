import pickle
import numpy as np
import pandas as pd

class GeneticSyndromeDataset:
    def __init__(self, file_path):
        """
        Inicializa a classe carregando os dados do arquivo pickle.
        """
        self.file_path = file_path
        self.data = self._load_data()
        self.df = self._flatten_data()

    def _load_data(self):
        """
        Carrega os dados do arquivo pickle.
        """
        try:
            with open(self.file_path, "rb") as f:
                data = pickle.load(f)
            print(f"Dados carregados com sucesso! Total de síndromes: {len(data)}")
            return data
        except Exception as e:
            print(f"Erro ao carregar os dados: {e}")
            return None

    def _flatten_data(self):
        """
        Converte a estrutura hierárquica em um DataFrame.
        """
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
        """
        Exibe estatísticas básicas do dataset.
        """
        if self.df is None:
            print("Os dados não foram carregados corretamente.")
            return

        print("\nResumo dos dados:")
        print(self.df.head())
        print("\nDistribuição de síndromes:")
        print(self.df["syndrome_id"].value_counts())
        print("\nValores nulos:")
        print(self.df.isnull().sum())
