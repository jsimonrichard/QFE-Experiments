from torch_geometric.datasets import TUDataset
from sklearn.decomposition import PCA
from config import Dataset
import argparse
import pickle
import os
import tempfile


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Dataset, required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp_dir:
        ds_temp = TUDataset(
            root=tmp_dir,
            name=args.dataset.value,
            use_node_attr=True
        )

        if ds_temp.num_features > 4:
            pca = PCA(n_components=4)
            pca.fit(ds_temp.x)

            os.makedirs(f"data/{args.dataset.value}", exist_ok=True)
            with open(f"data/{args.dataset.value}/pca.pkl", "wb") as f:
                pickle.dump(pca, f)