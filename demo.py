import numpy as np
from prone import prone

import time
from os.path import exists

# download the dataset if it doesn't exist
if not exists('datasets/bio_train.dat'):
    from tqdm import tqdm
    import requests
    import tarfile
    import os

    print("Dataset not found. Downloading.")

    url = 'https://kdd.org/cupfiles/KDDCupData/2004/data_kddcup04.tar.gz'
    filepath = 'datasets/kdd.tar.gz'

    response = requests.get(url, stream=True)
    
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024

    with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
        with open(filepath, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)

    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError("Could not download file")
    
    print("Extracting the file contents.")
    tar = tarfile.open(filepath)
    tar.extractall(path='./datasets', filter='data')
    tar.close()

    os.remove(filepath)

print("Reading dataset.")
X = np.genfromtxt('datasets/bio_train.dat', delimiter='\t')
k = 1000

print("Clustering the dataset.")
start = time.time()
centers, _ = prone(X, k)
end = time.time()

print(f'Found {k} centers in {end - start}s')
print(centers)
