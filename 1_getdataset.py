from kaggle.api.kaggle_api_extended import KaggleApi

# Step 1: Download dataset from Kaggle
api = KaggleApi()
api.authenticate()

# Dataset path on Kaggle
api.dataset_download_files('asaniczka/data-science-job-postings-and-skills', path='data/', unzip=True)

# for this to work you need to generate a token on your kaggle account under settings
# and the created kaggle.json should be stored on default location C:\Users\[your user]\.kaggle
print('Data downloaded and saved to ../data/')
