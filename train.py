import os
from dotenv import load_dotenv
import comet_ml
from comet_ml.integration.pytorch import log_model
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
from scipy.stats import bootstrap
from tqdm import tqdm

from config import Dataset, Embedding, ClassicalModel, Pooling, get_args, get_hparams_from_args
from model import build_model

def setup_comet_experiment(args):
    load_dotenv()

    # Prepare CometML
    comet_args = {
        "api_key": os.getenv("COMET_ML_API_KEY"),
        "project_name": "xu_qfe_qml",
        "workspace": "jsimonrichard",
        "disabled": args.dont_log
    }
    if args.exp_key:
        comet_args["previous_experiment"] = args.exp_key
        experiment_class = comet_ml.ExistingOfflineExperiment if args.offline else comet_ml.ExistingExperiment
    else:
        experiment_class = comet_ml.OfflineExperiment if args.offline else comet_ml.Experiment
    
    exp = experiment_class(**comet_args)

    if not args.exp_key:
        hparams = get_hparams_from_args(args)
        exp.log_parameters(hparams)
        exp.log_dataset_info(name=args.dataset.value)
    
    return exp


def train(model, device, train_loader, test_loader, args, exp, fold, model_checkpoint=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if model_checkpoint is not None:
        optimizer.load_state_dict(model_checkpoint["optimizer_state_dict"])

    model.train()

    start_epoch = 0 if model_checkpoint is None else model_checkpoint["epoch"] + 1

    min_loss = np.finfo(np.float32).max
    patience_cnt = 0

    for epoch in tqdm(range(start_epoch, args.epochs), initial=start_epoch):
        
        epoch_losses = []
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data.to(device))
            loss = F.nll_loss(out, data.y.to(device))
            loss.backward()
            epoch_losses.append(loss.item())
            optimizer.step()
        
        exp.log_metric(f"fold-{fold}/train/loss", sum(epoch_losses)/len(train_loader.dataset), epoch=epoch)
    
        acc, loss_test = test(model, device, test_loader)
        exp.log_metric(f"fold-{fold}/validation/accuracy", acc, epoch=epoch)
        exp.log_metric(f"fold-{fold}/validation/loss", loss_test, epoch=epoch)
        if loss_test < min_loss:
            min_loss = loss_test
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt == args.patience:
                exp.stop_early(epoch)
                break
        
        if epoch % 50 == 0:
            model_checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "fold": fold,
            }
            log_model(
                exp,
                model_checkpoint,
                model_name=f"fold-{fold}/model",
                metadata={"epoch": epoch, "fold": fold}
            )

        exp.log_epoch_end(epoch)
        
    return model
        

def test(model, device, data_loader):
    model.eval()
    correct = 0
    loss_test = 0.0
    for data in data_loader:
        out = model(data.to(device))
        correct += (out.argmax(dim=1) == data.y.to(device)).sum().item()
        loss_test += F.nll_loss(out, data.y.to(device)).item()
    return correct / len(data_loader.dataset), loss_test / len(data_loader.dataset)


def run_experiment(args):
    exp = setup_comet_experiment(args)

    # Reproducibility
    torch.manual_seed(args.seed)

    device = torch.device(args.device)

    # Get dataset
    ds = TUDataset(root="data", name=args.dataset.value)
    train_ds, test_ds = train_test_split(ds, test_size=0.1, stratify=ds.y)
    del test_ds # not used in this script; writen out for clarity

    k_fold_accuracies = []

    # Setup the K-Folded Experiments
    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    for fold, (train_index, val_index) in enumerate(skf.split(ds, ds.y)):
        print(f"Fold {fold}")
        
        # Split the dataset
        train_dataset = ds[train_index]
        val_dataset = ds[val_index]

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # Get model
        model = build_model(args, ds.num_features, ds.num_classes).to(device) 

        # Handle Restarts
        model_checkpoint = None
        if args.exp_key:
            model_checkpoint = exp.get_model_checkpoint(f"experiment://{args.exp_key}/fold-{fold}/model")
            if model_checkpoint is not None:
                model.load_state_dict(model_checkpoint["model_state_dict"])
        
        # Run Experiment
        model = train(model, device, train_loader, val_loader, args, exp, fold, model_checkpoint=model_checkpoint)

        # TEST
        # Since we will tune hyperparaters on separate runs,
        # we still need to use the validation set here.
        with exp.test():
            acc, _ = test(model, device, val_loader)
            exp.log_metric(f"fold-{fold}/test/accuracy", acc)
            k_fold_accuracies.append(acc)

    # Bootstrap the accuracies to get confidence interval
    res = bootstrap((k_fold_accuracies,), np.mean, confidence_level=0.95)
    m = (res.confidence_interval.low + res.confidence_interval.high)/2
    e = (res.confidence_interval.high - res.confidence_interval.low)/2
    print(f"Accuracy: {m} plus.minus {e}")
    exp.log_metric("accuracy", m)


if __name__ == "__main__":
    args = get_args()
    run_experiment(args)