from kaggle.api.kaggle_api_extended import KaggleApi

DATASET = "adityajn105/flickr30k"

api = KaggleApi()
api.authenticate()

files = api.dataset_list_files(DATASET)

print(type(files))
print("Number of files:", len(files.files))

for f in files.files[:20]:
    print(f.name, f.total_bytes)