from kaggle.api.kaggle_api_extended import KaggleApi
from config import KAGGLE_DATASET

api = KaggleApi()
api.authenticate()

files = api.dataset_list_files(KAGGLE_DATASET)

print(type(files))
print("Number of files:", len(files.files))

for f in files.files[:20]:
    print(f.name, f.total_bytes)