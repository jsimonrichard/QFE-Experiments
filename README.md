## Installing the virtual environment into jupyter

While the venv is activated:
```bash
python -m ipykernel install --user --name=qugcn_venv
```

Do not do this is using vscode. It will detect the python venv on its own; installing the kernel will just make it show up twice.