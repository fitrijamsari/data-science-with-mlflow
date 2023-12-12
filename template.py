import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")


project_name = "mlProject"


list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",
    "main.py",
    "app.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "templates/index.html",
    "test.py",
]


for filepath in list_of_files:
    filepath = Path(filepath)
    if not filepath.exists() or filepath.stat().st_size == 0:
        filedir = filepath.parent
        if filedir != Path():
            filedir.mkdir(parents=True, exist_ok=True)
            logging.info(f"Creating directory; {filedir} for the file: {filepath.name}")

        if not filepath.exists():
            filepath.touch()
            logging.info(f"Creating empty file: {filepath}")
        else:
            logging.info(f"{filepath.name} already exists")
    else:
        logging.info(f"{filepath.name} is already exists")
