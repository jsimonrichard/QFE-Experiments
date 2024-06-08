from torch_geometric.datasets import TUDataset
from sklearn.decomposition import PCA
from config import Dataset
import pickle
import os.path
import torch


def build_transform(pca):
    def transform(data):
        if pca is not None:
            data.x = torch.tensor(pca.transform(data.x), dtype=torch.float)
        return data
    return transform


def get_dataset(dataset: Dataset):
    pca_file = f"data/{dataset.value}/pca.pkl"
    transform = None
    if os.path.exists(pca_file):
        with open(f"data/{dataset.value}/pca.pkl", "rb") as f:
            pca = pickle.load(f)
            transform = build_transform(pca)

    ds = TUDataset(
        root="data",
        name=dataset.value,
        use_node_attr=True,
        transform=transform
    )
    
    return ds