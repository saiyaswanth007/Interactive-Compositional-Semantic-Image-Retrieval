from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "adityajn105/flickr30k"

api = KaggleApi()
api.authenticate()

files = api.dataset_list_files(DATASET)

print("\nALL DATASET FILES:\n")

for f in files.files:
    print(f.name)