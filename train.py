import os
import comet_ml
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import ShuffleSplit, StratifiedKFold
import numpy as np
from scipy.stats import bootstrap
from tqdm import tqdm
import string
import random
from fvcore.nn import FlopCountAnalysis
import optuna

from config import get_args, get_hparams_from_args, Embedding
from model import build_model
from dataset import get_dataset

def setup_comet_experiment(args):
    # Prepare CometML
    comet_ml.init()
    comet_args = {
        "project_name": "xu_qfe_qml",
        "workspace": "jsimonrichard",
    }
    if args.exp_key:
        comet_args["previous_experiment"] = args.exp_key
        experiment_class = comet_ml.ExistingOfflineExperiment if args.offline else comet_ml.ExistingExperiment
    else:
        experiment_class = comet_ml.OfflineExperiment if args.offline else comet_ml.Experiment
    
    exp = experiment_class(**comet_args)

    hparams = get_hparams_from_args(args)

    if args.exp_key:
        for key in hparams:
            assert hparams[key] == exp.get_parameter(key), f"Hparam {key} does not match."
    else:
        exp.log_parameters(hparams)
        exp.log_dataset_info(name=args.dataset.value)
    
    return exp

def log_model(run_key, model, optimizer, epoch, fold, patience_cnt, finished=False, exp=None):
    model_checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "fold": fold,
        "patience_cnt": patience_cnt, 
        "finished": finished
    }
    model_dir = f"./checkpoints/{run_key}"
    model_filename = f"{model_dir}/fold-{fold}-model.pth"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model_checkpoint, model_filename)

    if exp:
        exp.log_model("fold-{fold}-model", model_filename, overwrite=True)


def train(model, device, train_loader, test_loader, args, fold, exp_key=None, cml_exp=None, model_checkpoint=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if model_checkpoint is not None:
        optimizer.load_state_dict(model_checkpoint["optimizer_state_dict"])

    model.train()

    start_epoch = 0 if model_checkpoint is None else model_checkpoint["epoch"] + 1

    min_loss = np.finfo(np.float32).max
    patience_cnt = 0 if model_checkpoint is None else model_checkpoint["patience_cnt"]

    for epoch in tqdm(range(start_epoch, args.epochs), initial=start_epoch):
        
        epoch_losses = []
        for data in train_loader:
            optimizer.zero_grad()
            data = data.to(device)
            x, edge_index, batch = data.x, data.edge_index, data.batch
            out = model(x, edge_index, batch)
            loss = F.nll_loss(out, data.y.to(device))
            loss.backward()
            epoch_losses.append(loss.item())
            optimizer.step()
        
        if cml_exp:
            cml_exp.log_metric(f"fold-{fold}/train/loss", sum(epoch_losses)/len(train_loader.dataset), epoch=epoch)
    
        acc, loss_test, _, _ = test(model, device, test_loader)
        if cml_exp:
            cml_exp.log_metric(f"fold-{fold}/validation/accuracy", acc, epoch=epoch)
            cml_exp.log_metric(f"fold-{fold}/validation/loss", loss_test, epoch=epoch)
        
        if loss_test < min_loss:
            min_loss = loss_test
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt == args.patience:
                if cml_exp:
                    cml_exp.stop_early(epoch)
                break
        
        if epoch % 50 == 0 and not epoch == args.epochs-1 and exp_key:
            log_model(exp_key, model, optimizer, epoch, fold, patience_cnt, exp=cml_exp)

        if cml_exp:    
            cml_exp.log_epoch_end(epoch)
    
    if exp_key:
        log_model(exp_key, model, optimizer, epoch, fold, patience_cnt, finished=True, exp=cml_exp)

    return model
        

def test(model, device, data_loader):
    model.eval()
    correct = 0
    loss_test = 0.0
    predicted = torch.tensor([]).to(device)
    actual = torch.tensor([]).to(device)
    for data in data_loader:
        data = data.to(device)
        x, edge_index, batch = data.x, data.edge_index, data.batch
        out = model(x, edge_index, batch=batch)
        p = out.argmax(dim=1)
        correct += (p == data.y).sum().item()
        loss_test += F.nll_loss(out, data.y).item()

        predicted = torch.cat((predicted, p))
        actual = torch.cat((actual, data.y))

    acc = correct / len(data_loader.dataset)
    avg_loss = loss_test / len(data_loader.dataset)

    return acc, avg_loss, actual.to("cpu", dtype=torch.int), predicted.to("cpu", dtype=torch.int)

# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_experiment(args, trial=None):
    cml_exp = None
    if args.comet_ml:
        cml_exp = setup_comet_experiment(args)
        exp_key = cml_exp.get_key()
    elif args.exp_key:
        exp_key = args.exp_key
    else:
        exp_key = None
        while exp_key is None or os.path.exists(f"./checkpoints/{exp_key}"):
            exp_key = ''.join(random.choices(string.ascii_lowercase +
                                string.digits, k=16))

    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device)

    # Get dataset
    train_ds, test_ds = get_dataset(args)

    k_fold_accuracies = []

    # Setup the K-Folded Experiments
    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    for fold, (train_index, val_index) in enumerate(skf.split(train_ds, train_ds.y)):
        print(f"Fold {fold}")

        # Handle Restarts
        model_checkpoint = None
        if args.exp_key:
            model_checkpoint_name = f"./checkpoints/{args.exp_key}/fold-{fold}-model.pth"
            if os.path.exists(model_checkpoint_name):
                model_checkpoint = torch.load(model_checkpoint_name)
                epoch = model_checkpoint["epoch"]
                print(f"Model checkpoint at epoch {epoch} loaded")
        
        # Split the dataset
        train_dataset = train_ds[train_index]
        val_dataset = train_ds[val_index]

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        # Get model
        model = build_model(args, train_ds.num_features, train_ds.num_classes).to(device) 
        if model_checkpoint is not None:
            model.load_state_dict(model_checkpoint["model_state_dict"])
        
        # Run Experiment
        if not model_checkpoint or not model_checkpoint["finished"]:
            model = train(
                model, device, train_loader, val_loader,
                args, fold, exp_key=exp_key, cml_exp=cml_exp,
                model_checkpoint=model_checkpoint
            )
        else:
            print(f"Fold {fold} already finished. Skipping training.")

        # TEST
        # Since we will tune hyperparaters on separate runs,
        # we still need to use the validation set here.
        acc, _, actual, predicted = test(model, device, val_loader)

        if cml_exp:
            cml_exp.log_metric(f"fold-{fold}/test/accuracy", acc)
            cml_exp.log_confusion_matrix(actual, predicted, file_name=f"fold-{fold}-test-confusion_matrix.json")

        if trial:
            trial.report(acc, fold)
            if trial.should_prune():
                raise optuna.TrialPruned()

        k_fold_accuracies.append(acc)

    # Bootstrap the accuracies to get confidence interval
    res = bootstrap((k_fold_accuracies,), np.mean, confidence_level=0.95)
    m = (res.confidence_interval.low + res.confidence_interval.high)/2
    e = (res.confidence_interval.high - res.confidence_interval.low)/2
    print(f"Accuracy: {m} plus.minus {e}")

    param_count = count_parameters(model)
    print(f"Model has {param_count} parameters")

    if cml_exp:
        cml_exp.log_metric("accuracy", m)
        cml_exp.log_metric("accuracy_error", e)
        cml_exp.log_metric("param_count", param_count)

    return m, e, param_count


if __name__ == "__main__":
    args = get_args()
    run_experiment(args)