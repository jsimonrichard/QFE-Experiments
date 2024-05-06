from torch_geometric.datasets import TUDataset
from sklearn.model_selection import ShuffleSplit
from sklearn.decomposition import PCA

def get_dataset(args):
    ds = TUDataset(root="data", name=args.dataset.value, use_node_attr=True)
    
    if ds.num_features > 4:
        pca = PCA(n_components=4)
        ds.x = pca.fit_transform(ds.x)
    
    rs = ShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    train_index, test_index = next(rs.split(ds))

    return ds[train_index], ds[test_index]