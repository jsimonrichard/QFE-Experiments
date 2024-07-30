import os, sys
import argparse
import json
import argparse
from collections import defaultdict
from dotenv import load_dotenv
import numpy as np
from sklearn.model_selection import StratifiedKFold
from scipy.stats import bootstrap
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from dataset import get_dataset
from config import Dataset, Embedder, EnumAction
from eval import count_parameters, eval_
from model import load_model


def test_all(ds, model_dir, k_folds=5, seed=42, device="cuda"):
    data = defaultdict(
        lambda: defaultdict(
            lambda: {
                "accuracies": [],
                "param_counts": [],
            }
        )
    )

    folds = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

    for fold, (_train_index, test_index) in enumerate(folds.split(ds, ds.y)):
        print(f"--------------- Test Fold {fold} ---------------")

        # Get the training dataset
        test_ds = ds[test_index]
        test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False, num_workers=0)

        for embedder in Embedder:
            print(f"--------------- Embedder: {embedder} ---------------")
            for utility_fn in ["best_all", "best_acc", "low_params"]:
                print(f"--------------- Utility Function: {utility_fn} ---------------")
                model = load_model(
                    f"{model_dir}/embedder-{embedder.value}/utility-{utility_fn}/fold-{fold}-model.pth"
                ).to(device)
                acc, _, _, _ = eval_(model, device, test_loader)
                param_count = count_parameters(model)

                data[embedder][utility_fn]["accuracies"].append(acc)
                data[embedder][utility_fn]["param_counts"].append(param_count)

    return data


def calc_stats(data):
    stats = defaultdict(lambda: defaultdict(dict))

    for embedder, embedder_data in data.items():
        for utility_fn, exp_data in embedder_data.items():
            for stat, values in exp_data.items():
                res = bootstrap((values,), np.mean, confidence_level=0.95)
                m = (res.confidence_interval.low + res.confidence_interval.high) / 2
                e = (res.confidence_interval.high - res.confidence_interval.low) / 2
                stats[embedder][utility_fn][stat] = {
                    "mean": m,
                    "error": e,
                }

    return stats


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run all studies")
    parser.add_argument(
        "-d",
        "--dataset",
        action=EnumAction,
        enum_type=Dataset,
        required=True,
        help="Choose a dataset from: %(choices)s",
    )
    parser.add_argument("--k-folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument(
        "--model-dir",
        type=str,
        help="Study data path; default is ./study_outputs/dataset-{dataset_name}/",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Output path; default is ./study_outputs/dataset-{dataset_name}/results.json",
    )
    args = parser.parse_args()

    ds = get_dataset(args.dataset)
    model_dir = (
        args.model_dir
        if args.model_dir
        else f"./study_outputs/dataset-{args.dataset.value}/"
    )

    output_path = (
        args.output_path
        if args.output_path
        else f"./study_outputs/dataset-{args.dataset.value}/results.json"
    )

    raw_results = test_all(
        ds,
        model_dir,
        k_folds=args.k_folds,
        seed=args.seed,
        device=args.device,
    )
    stats = calc_stats(raw_results)

    results = {
        "raw_results": raw_results,
        "stats": stats,
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
