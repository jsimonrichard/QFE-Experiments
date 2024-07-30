import argparse
from sklearn.model_selection import StratifiedKFold
import torch
from fvcore.nn import FlopCountAnalysis
import torch.nn.functional as F

from config import Dataset, EnumAction
from dataset import get_dataset


def eval_(model, device, data_loader):
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

    return (
        acc,
        avg_loss,
        actual.to("cpu", dtype=torch.int),
        predicted.to("cpu", dtype=torch.int),
    )


# https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_flops(model, device, x, edge_index, batch=None):
    model.eval()
    return FlopCountAnalysis(
        model,
        inputs=(
            x.to(device),
            edge_index.to(device),
            None if batch is None else batch.to(device),
        ),
    ).total()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model")
    parser.add_argument(
        "-d",
        "--dataset",
        action=EnumAction,
        enum_type=Dataset,
        required=True,
        help="Choose a dataset from: %(choices)s",
    )
    parser.add_argument("-m", "--model", type=str, required=True, help="Model path")
    parser.add_argument("--k-folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--fold", type=int, default=0, help="Fold to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    args = parser.parse_args()

    ds = get_dataset(args.dataset)

    skf = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    train_index, test_index = list(skf.split(ds, ds.y))[args.fold]
    # train_ds = ds[train_index]
    test_ds = ds[test_index]

    model = torch.load(args.model).to(args.device)

    acc, _, _, _ = eval_(args, args.device, test_ds)
    print("Acc:", acc)

    param_count = count_parameters(model)
    print("Parameters:", param_count)
