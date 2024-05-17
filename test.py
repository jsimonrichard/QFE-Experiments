import comet_ml
import torch
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedKFold
import os, random, string
import numpy as np
from scipy.stats import bootstrap
from statsmodels.stats.proportion import proportion_confint

from train import setup_comet_experiment, train, test, count_parameters
from config import get_args, Embedding
from model import build_model
from dataset import get_dataset

def run_test(args):
    cml_exp = None
    if args.comet_ml:
        cml_exp = setup_comet_experiment(args)
        exp_key = cml_exp.get_key()
    elif args.exp_key:
        exp_key = args.exp_key
    else:
        exp_key = None
        while exp_key is None or os.path.exists(f"./checkpoints/{exp_key}"):
            exp_key = 'test_' + ''.join(random.choices(string.ascii_lowercase +
                                string.digits, k=16))

    # Reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device)

    # Get dataset
    train_ds, test_ds = get_dataset(args)

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    accuracies = []
    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    for fold, (train_index, val_index) in enumerate(skf.split(train_ds, train_ds.y)):
        
        train_loader = DataLoader(train_ds[train_index], batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(train_ds[val_index], batch_size=args.batch_size, shuffle=False)
        
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
            model, device, train_loader, val_loader,
            args, fold, exp_key=exp_key, cml_exp=cml_exp,
            model_checkpoint=model_checkpoint
        )

        acc, _, actual, predicted = test(model, device, test_loader)
        correct = (actual == predicted).sum()
        total = len(actual)

        if cml_exp:
            cml_exp.log_metric(f"fold-{fold}/test/accuracy", acc)
            cml_exp.log_confusion_matrix(actual, predicted, file_name=f"fold-{fold}-test-confusion_matrix.json")

        accuracies.append(acc)

        print(f"Fold {fold} Accuracy: {correct}/{total} ({acc})")
        
    # Bootstrap the accuracies to get confidence interval
    res = bootstrap((accuracies,), np.mean, confidence_level=0.95)
    acc = (res.confidence_interval.low + res.confidence_interval.high)/2
    e = (res.confidence_interval.high - res.confidence_interval.low)/2

    if cml_exp:
        cml_exp.log_metric("accuracy", acc)
        cml_exp.log_metric("accuracy_error", e)
        
    ci_low, ci_high = proportion_confint(acc*total, total, alpha=0.05, method="wilson")
    wilson_acc = (ci_low + ci_high)/2
    wilson_e = (ci_high - ci_low)/2

    param_count = count_parameters(model)

    if cml_exp:
        cml_exp.log_metric("accuracy_wilson_low", ci_low)
        cml_exp.log_metric("accuracy_wilson_high", ci_high)
        cml_exp.log_metric("accuracy_wilson", wilson_acc)
        cml_exp.log_metric("accuracy_wilson_error", wilson_e)
        cml_exp.log_metric("param_count", param_count)

    return acc, e, wilson_acc, wilson_e, param_count

if __name__ == "__main__":
    args = get_args()
    acc, error, wilson_acc, wilson_e, param_count = run_test(args)
    print("Acc:", acc)
    print("Error:", error)
    print("Wilson Acc:", wilson_acc)
    print("Wilson Error:", wilson_e)
    print("Parameters:", param_count)
