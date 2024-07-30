from copy import deepcopy
import os, sys
import argparse
import json

import torch
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from config import Dataset, EnumAction, Embedder, ClassicalModel, Pooling, gen_args
from dataset import get_dataset
from model import build_model, save_model
from train import train


def gen_args_from_params_dict(dataset, embedder, params, seed=42, device="cuda"):
    params = deepcopy(params)
    model = ClassicalModel[params["model"]]
    del params["model"]
    pooling = Pooling[params["pooling"].upper()]
    del params["pooling"]

    args = gen_args(
        dataset,
        embedder,
        model,
        pooling,
        batch_size=1024,
        comet_ml=True,
        device=device,
        seed=seed,
        **params,
    )

    return args


def run_training(args, train_ds, min_epochs: int, max_tries: int):
    model = build_model(args, train_ds.num_features, train_ds.num_classes).to(
        args.device
    )

    sss = StratifiedShuffleSplit(test_size=0.2, random_state=args.seed)
    train_index, val_index = next(sss.split(train_ds, train_ds.y))

    train_loader = DataLoader(
        train_ds[train_index], batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        train_ds[val_index], batch_size=args.batch_size, shuffle=False
    )

    # Sometimes training fails drastically, so we'll try up to 5 times based on validation performance
    epoch = 0
    tries = 0
    while epoch < min_epochs and tries < max_tries:
        model, epoch = train(
            model,
            args.device,
            train_loader,
            val_loader,
            args,
            None,
        )
        tries += 1

    return model


def train_all(
    study_data,
    ds,
    dataset_name: Dataset,
    min_epochs: int,
    max_tries: int,
    output_dir: str,
    seed=42,
    device="cuda",
):
    # Split dataset in to train and test folds
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # Outer cross-validation
    for fold, (train_index, _test_index) in enumerate(folds.split(ds, ds.y)):
        print(f"--------------- Train Fold {fold} ---------------")

        # Get the training dataset
        train_ds = ds[train_index]

        # Run Optuna studies for each embedder variation
        for embedder in Embedder:
            print(f"--------------- Embedder: {embedder} ---------------")

            # Run the study
            for utility_fn, trials in study_data[embedder]["best_trials"].items():
                trial = trials[fold]
                print(f"--------------- Utility Function: {utility_fn} ---------------")

                args = gen_args_from_params_dict(
                    dataset_name, embedder, trial["params"], seed=seed, device=device
                )
                model = run_training(args, train_ds, min_epochs, max_tries)

                os.makedirs(
                    f"{output_dir}/embedder-{embedder.value}/utility-{utility_fn}",
                    exist_ok=True,
                )
                save_model(
                    args,
                    train_ds.num_features,
                    train_ds.num_classes,
                    model,
                    f"{output_dir}/embedder-{embedder.value}/utility-{utility_fn}/fold-{fold}-model.pth",
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train all of the best models with parameters from studies"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        action=EnumAction,
        enum_type=Dataset,
        required=True,
        help="Choose a dataset from: %(choices)s",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument(
        "--min-epochs", type=int, default=60, help="Minimum number of epochs"
    )
    parser.add_argument(
        "--max-tries", type=int, default=5, help="Maximum number of tries"
    )
    parser.add_argument(
        "--study-data",
        type=str,
        help="Study data path; default is ./study_outputs/dataset-{dataset_name}/study_data.json",
    )
    parser.add_argument(
        "--model-output-dir",
        type=str,
        help="Save the final models under this path; default is ./study_outputs/dataset-{dataset_name}/",
    )
    args = parser.parse_args()

    ds = get_dataset(args.dataset)
    study_data_path = (
        args.study_data
        if args.study_data
        else f"./study_outputs/dataset-{args.dataset.value}/study_data.json"
    )
    output_dir = (
        args.model_output_dir
        if args.model_output_dir
        else f"./study_outputs/dataset-{args.dataset.value}/"
    )

    with open(study_data_path) as f:
        study_data = json.load(f)

    train_all(
        study_data,
        ds,
        args.dataset,
        args.min_epochs,
        args.max_tries,
        output_dir,
        seed=args.seed,
        device=args.device,
    )
