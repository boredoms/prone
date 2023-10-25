# Fast k-Means Clustering

This repository contains the code for the subsision.

Building uses the conan build system, version 1.x.x and CMake.

Builds can be configured using `ccmake -S . -B build`, `cmake --build build` then builds the utility, with the binary placed in ./build/src/corset

The utility's usage is
- corset cluster filename num_centers: Clusters a csv dataset into k centers
- corset coreset filename num_centers size: Create a coreset for k means clustering of size size
- corset score filename centroids_file: Compute the clustering cost of the centroids on the dataset
- corset benchmark clustering runs file: Cluster the file using the different algorithms (experimental setup of the paper's section 3.2)
- corset benchmark coreset runs file: Create runs many coresets using the different algorithms (experimental setup of the paper's section 3.1)

## Data Analysis

The ipython notebook contains all data analysis and clustering using scipy. Additionally, the python script score_clusters.py can be invoked on a coreset to produce a set of centroids that can then be scored on the original dataset using the score utility of the corset binary.
