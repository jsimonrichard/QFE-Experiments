import comet_ml  # import first so that data from torch and sklearn can be automatically logged
import os, sys
from dotenv import load_dotenv
import argparse
from sklearn.model_selection import StratifiedKFold
import optuna
from optuna.storages import RDBStorage
import numpy as np
from collections import defaultdict
from scipy.stats import bootstrap
import json
from copy import deepcopy

sys.path.insert(0, os.path.dirname(__file__))
from config import EnumAction, Dataset, Embedder, ClassicalModel, Pooling, gen_args
from dataset import get_dataset
from test_ import run_test


def get_best_utility_trials(best_trials):
    accuracies = np.array([t.values[0] for t in best_trials])
    errors = np.array([t.values[1] for t in best_trials])
    params = np.array([t.values[2] for t in best_trials])

    # Get distributions
    acc_mean = np.mean(accuracies)
    acc_std = np.std(accuracies)
    err_mean = np.mean(errors)
    err_std = np.std(errors)
    param_mean = np.mean(params)
    param_std = np.std(params)

    def gen_utility(acc_weight, err_weight, param_weight):
        def utility(trial):
            acc_z = (trial.values[0] - acc_mean) / acc_std
            err_z = -(trial.values[1] - err_mean) / err_std
            param_z = -(trial.values[2] - param_mean) / param_std
            return acc_weight * acc_z + err_weight * err_z + param_weight * param_z

        return utility

    best_all = max(best_trials, key=gen_utility(1, 1, 1))
    best_acc = max(best_trials, key=gen_utility(1, 0, 0))
    low_params = max(best_trials, key=gen_utility(1, 0, 3))

    return {"best_all": best_all, "best_acc": best_acc, "low_params": low_params}


def gen_args_from_optuna_params(dataset, embedder, params):
    params = deepcopy(params)
    model = ClassicalModel[params["model"]]
    del params["model"]
    pooling = Pooling[params["pooling"].upper()]
    del params["pooling"]

    args = gen_args(
        dataset, embedder, model, pooling, batch_size=1024, comet_ml=True, **params
    )

    return args


def run_tests(storage, ds):
    best_trials = defaultdict(lambda: defaultdict(list))
    accuracies = defaultdict(lambda: defaultdict(list))
    exps = defaultdict(lambda: defaultdict(list))

    # Split dataset in to train and test folds
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    # Outer cross-validation
    for fold, (train_index, test_index) in enumerate(folds.split(ds, ds.y)):
        print(f"--------------- Train/Test Fold {fold} ---------------")

        # Get the training dataset
        train_ds = ds[train_index]
        test_ds = ds[test_index]

        # Run Optuna studies for each embedder variation
        for embedder in Embedder:
            print(f"--------------- Embedder: {embedder} ---------------")

            # Run the study
            study = optuna.load_study(
                study_name=f"dataset-{args.dataset.value}/embedder-{embedder.value}/fold-{fold}",
                storage=storage,
            )
            for name, trial in get_best_utility_trials(study.best_trials).items():
                print(f"Best {name} validation results: {trial.values}")
                print(f"Best {name} params: {trial.params}")

                best_trials[embedder][name].append(
                    {
                        "number": trial.number,
                        "validation_results": trial.values,
                        "params": trial.params,
                    }
                )

                test_args = gen_args_from_optuna_params(
                    args.dataset, embedder, trial.params
                )
                acc, _, exp_key = run_test(
                    test_args,
                    train_ds,
                    test_ds,
                    exp_name=f"dataset-{args.dataset.value}/embedder-{embedder.value}/utility-{name}/fold-{fold}",
                )

                accuracies[embedder][name].append(acc)
                exps[embedder][name].append(exp_key)

                print(f"Test Accuracy: {acc}")
                print(f"Experiment Key: {exp_key}")
                print()

    return best_trials, accuracies, exps


def calc_results(accuracies):
    results = defaultdict(dict)

    for embedder, embedder_accs in accuracies.items():
        for name, accs in embedder_accs.items():
            res = bootstrap((accs,), np.mean, confidence_level=0.95)
            acc = (res.confidence_interval.low + res.confidence_interval.high) / 2
            e = (res.confidence_interval.high - res.confidence_interval.low) / 2

            results[embedder][name] = {"accuracy": acc, "error": e}

    return results


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run all studies")
    parser.add_argument(
        "--dataset",
        action=EnumAction,
        enum_type=Dataset,
        required=True,
        help="Choose a dataset from: %(choices)s",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    ds = get_dataset(args.dataset)

    storage = RDBStorage(url=os.getenv("OPTUNA_DB"))

    best_trials, accuracies, exps = run_tests(storage, ds)
    print("\n Calculating results...")
    results = calc_results(accuracies)

    # Save the results
    data = {
        "best_trials": best_trials,
        "accuracies": accuracies,
        "exps": exps,
        "results": results,
    }
    os.makedirs(f"study_outputs/dataset-{args.dataset}", exist_ok=True)
    results_file = f"study_outputs/dataset-{args.dataset}/results.json"
    with open(results_file, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Results saved to {results_file}")
