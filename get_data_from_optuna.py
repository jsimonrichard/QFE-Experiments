from collections import defaultdict
import os, sys
from dotenv import load_dotenv
import argparse
import optuna
from optuna.storages import RDBStorage
from optuna.trial import TrialState
import numpy as np
import json

sys.path.insert(0, os.path.dirname(__file__))
from config import EnumAction, Dataset, Embedder, ClassicalModel, Pooling, gen_args
from dataset import get_dataset


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


def get_trial_data(trial):
    return {
        "number": trial.number,
        "validation_results": {
            "accuracy": trial.values[0],
            "error": trial.values[1],
            "param_count": trial.values[2],
        },
        "params": trial.params,
        "duration": str(trial.datetime_complete - trial.datetime_start),
    }


def get_data(dataset, storage, k_folds=5):
    data = {}
    for embedder in Embedder:
        data[embedder] = {
            "best_trials": {
                "best_all": [],
                "best_acc": [],
                "low_params": [],
            },
            "pareto_front": [],
            "all_trials": [],
        }
        for fold in range(k_folds):
            study = optuna.load_study(
                study_name=f"dataset-{dataset.value}/embedder-{embedder.value}/fold-{fold}",
                storage=storage,
            )
            trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
            data[embedder]["all_trials"].append(list(map(get_trial_data, trials)))
            data[embedder]["pareto_front"].append(
                list(map(get_trial_data, study.best_trials))
            )

            best_trials = get_best_utility_trials(study.best_trials)
            for utility_fn, trial in best_trials.items():
                data[embedder]["best_trials"][utility_fn].append(get_trial_data(trial))

    return data


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Extract data (including hyperparameters) from Optuna"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        action=EnumAction,
        enum_type=Dataset,
        required=True,
        help="Choose a dataset from: %(choices)s",
    )
    parser.add_argument("--k-folds", type=int, default=5, help="Number of folds")
    parser.add_argument("-o", "--output", type=str, help="Output file")
    args = parser.parse_args()

    output_path = (
        args.output
        if args.output
        else (f"./study_outputs/dataset-{args.dataset.value}/study_data.json")
    )
    storage = RDBStorage(url=os.getenv("OPTUNA_DB"))

    data = get_data(args.dataset, storage, k_folds=args.k_folds)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
