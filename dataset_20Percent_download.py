from kaggle.api.kaggle_api_extended import KaggleApi
from config import KAGGLE_DATASET

api = KaggleApi()
api.authenticate()

files = api.dataset_list_files(KAGGLE_DATASET)

print("\nALL DATASET FILES:\n")

for f in files.files:
    print(f.name)