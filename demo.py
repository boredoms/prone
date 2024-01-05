import numpy as np
from prone import prone
import time

X = np.genfromtxt('experiments/datasets/bio_train.dat', delimiter=',')
k = 1000

start = time.time()
centers, _ = prone(X, k)
end = time.time()

print(f"Found {k} centers in {end - start}s")
print(centers)