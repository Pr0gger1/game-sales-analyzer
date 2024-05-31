import os

DATASET_FOLDER: str = os.path.join(os.path.dirname(__file__), "data")
DATASET_FILENAME: str = "vgchartz-2024.csv"
DATASET_FULLPATH: str = os.path.join(DATASET_FOLDER, DATASET_FILENAME)

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
