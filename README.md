# Hybrid Quantum or Purely Classical? Assessing the Utility of Quantum Feature Embeddings

Paper: coming soon...

Cite: coming soon...

### Results

The raw results data have been saved (and checked into `git`) at the paths following this form: `./study_outputs/<dataset name>/results.json`.

## Setting up the virtual environment

First, create a virtual environment and install the requirements:

```bash
python -m venv qfe_exp_venv
source qfe_exp_venv/bin/activate
pip install -r requirements.txt
```

If you find that a dependency is missing from `requirements.txt`, please open an issue or PR.

### Installing the virtual environment into Jupyter

To use the virtual environment in Jupyter/JupyterLab (not VSCode), run the following command while the venv is activated:
```bash
python -m ipykernel install --user --name=qugcn_venv
```

Do not do this if using VSCode; it will detect the python venv on its own. Installing the kernel will just make it show up twice.

## Data

This project uses the `PROTEINS` and `ENZYMES` datasets from http://graphlearning.io. Since both of these are included in the [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/datasets.html) library, you do not have manually download them. Instead, just run any of the scripts in this repo and the datasets will be automatically downloaded into the `data/` directory.

## Training

### Running individual experiments

To run individual experiments, you may use the `train.py` script. However, I would recommend skipping to the section on "running individual experiments with a test included."

Start by running
```bash
python train.py --help
```
to see the available options.

For example, to run a single experiment with the default hyperparameters (the hyperparameters below do not have defaults), run the following command:
```bash
python train.py --dataset PROTEINS --embedder QFE-exp --model GCN --pooling sum
```

### Running Hyperparameter Tuning

A single script is used to run all of the Optuna studies used for tuning hyperparameters on a given dataset: `run_strudies.py`. However, before you start, make sure to start the database Optuna requires for running studies in parallel:
```bash
docker compose up -d
```

Then, run the script:
```bash
python run_studies.py --dataset PROTEINS
```

If you want to run multiple studies in parallel, simply open another terminal and run the same command.

To see optuna's current progress, open the dashboard with the following command:
```bash
./optuna_dashboard.sh
```

## Testing

### Running individual experiments with a test included

To run individual tests, you may use the `test.py` script. Start by running
```bash
python test.py --help
```
to see the available options.

You may notice that the options for `test.py` are the same as those for `train.py`. Because training a single model is fast, I didn't implement a load-from-file option for `test.py`. So, when ever you run `test.py`, it will first train the model then test it.

For example, to run a single test with the default hyperparameters (the hyperparameters below do not have defaults), run the following command:
```bash
python test.py --dataset PROTEINS --embedder QFE-exp --model GCN --pooling sum
```

### Running all tests

To run all tests for the PROTEINS dataset, run the following command:
```bash
python test_from_studies.py --dataset PROTEINS
```
Once the tests are complete, the results will be saved as a JSON file to `./study_outputs/dataset-PROTEINS/results.json`.

To run all tests for the ENZYMES dataset, replace `PROTEINS` with `ENZYMES` in the command above.


### Reproducibility Issues

I was unable to achieve completely reproducible runs on my machine; your milage may vary.