import os, sys
import argparse
from sklearn.model_selection import StratifiedKFold
from dotenv import load_dotenv
import optuna
from optuna.storages import RDBStorage, RetryFailedTrialCallback
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
import torch
import gc

from config import EnumAction, Dataset, Embedder, ClassicalModel, Pooling, gen_args
from dataset import get_dataset
from train import run_experiment

TRIALS_PER_FOLD = 200


def objective(
    trial,
    dataset: Dataset,
    embedder: Embedder,
    fold,
    train_ds,
    inner_fold=5,
    device="cuda",
    seed=42,
):

    log_folder = f"./study_outputs/dataset-{dataset.value}/embedder-{embedder.value}"
    log_prefix = f"{log_folder}/fold-{fold}"

    os.makedirs(log_folder, exist_ok=True)

    # Redirect stdout and stderr to files
    sys.stdout = open(log_prefix + ".out", "a", buffering=1)
    sys.stderr = open(log_prefix + "_error.out", "a", buffering=1)

    pooling = trial.suggest_categorical(
        "pooling", [Pooling.SUM, Pooling.MEAN, Pooling.MAX]
    )
    # Sometimes the pooling is returned as a string for some reason
    if type(pooling) == str:
        pooling = Pooling[pooling.upper()]

    model = trial.suggest_categorical("model", list(ClassicalModel))
    # Sometimes the model is returned as a string for some reason
    if type(model) == str:
        model = ClassicalModel[model]

    if embedder.value.split("-")[0] == "QFE":
        qfe_layers = trial.suggest_int("qfe_layers", 1, 4)
    else:
        qfe_layers = 2

    # Generate arguments
    args = gen_args(
        dataset=dataset,
        embedder=embedder,
        model=model,
        pooling=pooling,
        qfe_layers=qfe_layers,
        layers=trial.suggest_int("layers", 1, 16),
        hidden=trial.suggest_int("hidden", 16, 256),
        dropout=trial.suggest_float("dropout", 0.0, 0.5),
        lr=trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        batch_size=1024,
        device=device,
        seed=seed,
        k_folds=inner_fold,
    )

    result = run_experiment(args, train_ds, save_checkpoints=False)

    torch.cuda.empty_cache()
    gc.collect()

    return result


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
    parser.add_argument("--k-folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    ds = get_dataset(args.dataset)

    storage = RDBStorage(
        url=os.getenv("OPTUNA_DB"),
        heartbeat_interval=60,
        grace_period=120,
        failed_trial_callback=RetryFailedTrialCallback(max_retry=3),
    )
    max_trials_callback = MaxTrialsCallback(
        TRIALS_PER_FOLD, states=(TrialState.COMPLETE,)
    )

    # Split dataset in to train and test folds
    folds = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

    # Outer cross-validation
    for fold, (train_index, _test_index) in enumerate(folds.split(ds, ds.y)):
        print(f"--------------- Train/Test Fold {fold} ---------------")

        # Get the training dataset
        train_ds = ds[train_index]

        # Run Optuna studies for each embedder variation
        for embedder in Embedder:
            print(f"--------------- Embedder: {embedder} ---------------")

            # Run the study
            study = optuna.create_study(
                study_name=f"dataset-{args.dataset.value}/embedder-{embedder.value}/fold-{fold}",
                storage=storage,
                load_if_exists=True,
                directions=["maximize", "minimize", "minimize"],
            )

            current_trials = study.get_trials(
                deepcopy=False, states=max_trials_callback._states
            )
            if len(current_trials) >= TRIALS_PER_FOLD:
                print("Already completed. Skipping.")
                continue

            study.optimize(
                lambda trial: objective(
                    trial,
                    args.dataset,
                    embedder,
                    fold,
                    train_ds,
                    inner_fold=args.k_folds,
                    device=args.device,
                    seed=args.seed,
                ),
                n_trials=TRIALS_PER_FOLD,
                callbacks=[max_trials_callback],
            )
