## Installing the virtual environment into jupyter

While the venv is activated:
```bash
python -m ipykernel install --user --name=qugcn_venv
```

Do not do this if using vscode. It will detect the python venv on its own; installing the kernel will just make it show up twice.

## Running Optuna with Docker Compose

```
docker compose up -d
./optuna_dashboard.sh
```

## Reproducibility Issues

I was unable to achieve completely reproducible runs on my machine, but your milage may vary.