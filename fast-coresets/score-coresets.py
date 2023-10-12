#!/usr/bin/env python3

from os import walk
from sklearn.cluster import k_means
import numpy as np

def load_coreset(filename):
    csv = np.genfromtxt(filename, delimiter=",")
    n, d = csv.shape
    [points, weights] = np.split(csv, [d-1], axis=1)
    return points, weights

def cluster_coreset_and_export(filename, num_centers, num_runs = 5):
    points, weights = load_coreset(filename)
    weights = weights.flatten()
    for i in range(num_runs):
        print(f"Run {i}")
        centroids, _, _ = k_means(points, num_centers,
                                           sample_weight=weights, n_init='auto')

        stem = filename.split('.')[0]
        stem = stem.split('/')[-1]
        np.savetxt("clustered-coresets/" + stem + f"-clustered-{i}.csv", centroids, delimiter=',')

coreset_files = []
for (dirpath, dirnames, filenames) in walk("coresets/"):
    coreset_files.extend(filenames)
    break

count = 1
for filename in coreset_files:
    print(filename + f"({count} of {len(coreset_files)})")
    count = count + 1
    num_clusters = int(filename.split("-")[2])
    num_samples = int(filename.split("-")[3])
    if (num_samples >= num_clusters):
        cluster_coreset_and_export("coresets/" + filename, num_clusters)
