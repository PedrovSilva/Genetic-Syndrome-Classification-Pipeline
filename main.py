from dataset import GeneticSyndromeDataset

def data_visualization():
    file_path = "data/mini_gm_public_v0.1.p"

    print("Iniciando o carregamento dos dados...")
    dataset = GeneticSyndromeDataset(file_path)

    print("\nExibindo resumo dos dados:")
    dataset.get_summary()

    print("\nGerando visualização dos embeddings com t-SNE...")
    dataset.plot_tsne()

if __name__ == "__main__":
    data_visualization()
