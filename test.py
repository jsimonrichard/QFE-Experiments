import comet_ml
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import os, random, string
import numpy as np
from scipy.stats import bootstrap
from statsmodels.stats.proportion import proportion_confint

from train import setup_comet_experiment, train, test, count_parameters
from config import get_args, Embedder
from model import build_model
from dataset import get_dataset


def run_test(args, train_ds, test_ds, exp_name=None):
    cml_exp = None
    if args.comet_ml:
        cml_exp = setup_comet_experiment(args, exp_name=exp_name)
        exp_key = cml_exp.get_key()
    elif args.exp_key:
        exp_key = args.exp_key
    else:
        exp_key = None
        while exp_key is None or os.path.exists(f"./checkpoints/{exp_key}"):
            exp_key = "".join(
                random.choices(string.ascii_lowercase + string.digits, k=16)
            )

    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device)

    sss = StratifiedShuffleSplit(test_size=0.2, random_state=args.seed)
    train_index, val_index = next(sss.split(train_ds, train_ds.y))

    train_loader = DataLoader(
        train_ds[train_index], batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        train_ds[val_index], batch_size=args.batch_size, shuffle=False
    )
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    # Handle Restarts
    model_checkpoint = None
    if args.exp_key:
        model_checkpoint_name = f"./checkpoints/{args.exp_key}/fold-0-model.pth"
        if os.path.exists(model_checkpoint_name):
            model_checkpoint = torch.load(model_checkpoint_name)
            epoch = model_checkpoint["epoch"]
            print(f"Model checkpoint at epoch {epoch} loaded")

    model = build_model(args, train_ds.num_features, train_ds.num_classes).to(device)
    if model_checkpoint is not None:
        model.load_state_dict(model_checkpoint["model_state_dict"])

    model = train(
        model,
        device,
        train_loader,
        val_loader,
        args,
        0,
        exp_key=exp_key,
        cml_exp=cml_exp,
        model_checkpoint=model_checkpoint,
    )

    acc, _, actual, predicted = test(model, device, test_loader)

    if cml_exp:
        cml_exp.log_metric(f"test/accuracy", acc)
        cml_exp.log_confusion_matrix(
            actual, predicted, file_name=f"test-confusion_matrix.json"
        )

    param_count = count_parameters(model)

    if cml_exp:
        cml_exp.log_metric("accuracy", acc)
        cml_exp.log_metric("param_count", param_count)

    return acc, param_count, exp_key, model


if __name__ == "__main__":
    args = get_args()

    ds = get_dataset(args.dataset)

    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    train_index, test_index = next(skf.split(ds, ds.y))
    train_ds = ds[train_index]
    test_ds = ds[test_index]

    acc, param_count, _ = run_test(args, train_ds, test_ds)
    print("Acc:", acc)
    print("Parameters:", param_count)
