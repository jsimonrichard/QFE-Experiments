import os, sys
from tqdm import tqdm
import logging
import argparse
import optuna
from dotenv import load_dotenv

from config import Dataset, Embedding, ClassicalModel, Pooling, gen_args, EnumAction
from train import run_experiment


def objective(trial, pre_args, count_flops=False):


    process_name = f"process_out/{pre_args.dataset.value}_{pre_args.embedding.value}"
    
    # Redirect stdout and stderr to files
    sys.stdout = open(process_name + ".out", "a", buffering=1)
    sys.stderr = open(process_name + "_error.out", "a", buffering=1)

    # # Redirect comet_ml logging
    # file_handler = logging.FileHandler(process_name + ".log")
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # file_handler.setFormatter(formatter)
    # for key, logger in logging.root.manager.loggerDict.items():
    #     if key.startswith('comet_ml') and type(logger) == logging.Logger:
    #         for handler in logger.handlers[:]:  # Iterate over a copy of the handler list
    #             logger.removeHandler(handler)
    #         logger.addHandler(file_handler)
    #         logger.setLevel(logging.INFO)

    if pre_args.hgp_sl:
        pooling_choices = [Pooling.HGPSL]
    else:
        pooling_choices = [Pooling.SUM, Pooling.MEAN, Pooling.MAX]

    model = trial.suggest_categorical("model", list(ClassicalModel))
    pooling = trial.suggest_categorical("pooling", pooling_choices)

    if type(model) == str:
        model = ClassicalModel[model]
    if type(pooling) == str:
        pooling = Pooling[pooling.replace("-", "").upper()]

    qfe_layers = 2
    if pre_args.embedding.value.split("-")[0] == "QFE":
        qfe_layers = trial.suggest_int("qml_layers", 1, 4)

    args = gen_args(
        dataset=pre_args.dataset,
        embedding=pre_args.embedding,
        model=model,
        pooling=pooling,
        layers=trial.suggest_int("layers", 1, 16),
        qfe_layers=qfe_layers,
        hidden=trial.suggest_int("hidden", 16, 256),
        dropout=trial.suggest_float("dropout", 0.0, 0.5),
        lr=trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        batch_size=trial.suggest_categorical("batch_size", [512, 1024, 2048])
    )
    return run_experiment(args, count_flops=count_flops)

if __name__ == "__main__":
    load_dotenv() 

    parser = argparse.ArgumentParser(description="Tune hyperparameters with Optima")
    parser.add_argument('--dataset', action=EnumAction, enum_type=Dataset, required=True,
                        help='Choose a dataset from: %(choices)s')
    parser.add_argument('--embedding', action=EnumAction, enum_type=Embedding, required=True,
                        help='Choose a dataset from: %(choices)s')
    parser.add_argument('--hgp-sl', action="store_true", help='Use HGP-SL model')
    args = parser.parse_args()

    os.makedirs("optuna_studies", exist_ok=True)
    os.makedirs("process_out", exist_ok=True)

    directions = ["maximize", "minimize"]
    if args.embedding.value.split("-")[0] != "QFE":
        directions.append("minimize")

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = f"{args.dataset.value}_{args.embedding.value}_{'hgp-sl' if args.hgp_sl else 'simple-pooling'}"
    study = optuna.create_study(
        study_name=study_name,
        storage=os.getenv("OPTUNA_DB"),
        load_if_exists=True,
        directions=directions, 
        pruner=optuna.pruners.MedianPruner( # Will operate on the accuracies of each fold
            n_startup_trials=10,
            n_warmup_steps=4,
            interval_steps=1
        )
    )

    count_flops = args.embedding.value.split("-")[0] != "QFE"

    study.optimize(lambda trial: objective(trial, args, count_flops), n_trials=500, n_jobs=1)
