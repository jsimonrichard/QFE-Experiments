import argparse
from enum import Enum
from copy import deepcopy


class Dataset(str, Enum):
    PROTEINS = "PROTEINS"
    ENZYMES = "ENZYMES"


class Embedder(str, Enum):
    QFE_EXP = "QFE-exp"
    QFE_PROBS = "QFE-probs"
    MLP_2_D = "MLP-2^D"
    MLP_D = "MLP-D"
    NONE = "none"


class ClassicalModel(str, Enum):
    GCN = "GCN"
    GraphConv = "GraphConv"
    GraphSAGE = "GraphSAGE"
    GAT = "GAT"


class Pooling(str, Enum):
    SUM = "sum"
    MEAN = "mean"
    MAX = "max"


class EnumAction(argparse.Action):
    def __init__(self, enum_type, **kwargs):
        self._enum_type = enum_type
        kwargs["choices"] = [e.value for e in enum_type]  # List of enum values
        super().__init__(**kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        enum_value = self._enum_type(values)
        setattr(namespace, self.dest, enum_value)


def get_parser():
    parser = argparse.ArgumentParser(description="Graph Classification")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument(
        "--dataset",
        action=EnumAction,
        enum_type=Dataset,
        required=True,
        help="Choose a dataset from: %(choices)s",
    )
    parser.add_argument(
        "--k-folds",
        type=int,
        default=5,
        help="Number of folds for stratified k-fold cross-validation",
    )
    parser.add_argument(
        "--embedder",
        action=EnumAction,
        enum_type=Embedder,
        required=True,
        help="Choose a dataset from: %(choices)s",
    )
    parser.add_argument(
        "--qfe-layers",
        type=int,
        default=2,
        help="Number of layers to use in the QFE circuit",
    )
    parser.add_argument(
        "--model",
        action=EnumAction,
        enum_type=ClassicalModel,
        required=True,
        help="Choose a dataset from: %(choices)s",
    )
    parser.add_argument("--layers", type=int, default=8, help="Number of layers")
    parser.add_argument("--hidden", type=int, default=64, help="Number of hidden units")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--pooling",
        action=EnumAction,
        enum_type=Pooling,
        required=True,
        help="Choose a dataset from: %(choices)s",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=30, help="Patience")
    parser.add_argument("--batch-size", type=int, default=2048, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--exp-key", type=str, help="Output directory")
    parser.add_argument("--start-from", type=str, help="Start from a checkpoint")
    parser.add_argument("--comet-ml", action="store_true", help="Log to CometML")
    parser.add_argument("--offline", action="store_true", help="Use CometML offline")

    return parser


def get_args():
    parser = get_parser()
    args = parser.parse_args()
    return args


def gen_args(
    dataset: Dataset,
    embedder: Embedder,
    model: ClassicalModel,
    pooling: Pooling,
    layers: int = 8,
    qfe_layers: int = 2,
    hidden: int = 64,
    dropout: float = 0.1,
    lr: float = 0.001,
    weight_decay: float = 5e-4,
    batch_size: int = 2048,
    comet_ml=False,
):
    parser = get_parser()
    cli_args = [
        "--dataset",
        dataset.value,
        "--embedder",
        embedder.value,
        "--model",
        model.value,
        "--pooling",
        pooling.value,
        "--layers",
        str(layers),
        "--hidden",
        str(hidden),
        "--dropout",
        str(dropout),
        "--lr",
        str(lr),
        "--weight-decay",
        str(weight_decay),
        "--batch-size",
        str(batch_size),
        "--qfe-layers",
        str(qfe_layers),
    ]
    if comet_ml:
        cli_args.append("--comet-ml")

    args = parser.parse_args(cli_args)
    return args


def get_hparams_from_args(args):
    d = deepcopy(vars(args))
    for key, value in d.items():
        if isinstance(value, Enum):
            d[key] = value.value
    return d


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    print(get_hparams_from_args(args))
