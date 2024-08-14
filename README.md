# `PRONE` (PRojected ONE-dimensional clustering)

This repository contains a ready to use implementation of the `PRONE` algorithm from the paper "Simple, Scalable and Effective Clustering via One-Dimensional Projections". This is a reference implementation published alongside our paper.

It can be installed and used in python using `pip`. Installing can be done using `pip install .` in the prone directory. The dependencies are `numpy` and `cython`. To run the tests for the C++ code, you need to have `CMake` installed. If you want to install the packages for the demo, run `pip install .[demo]`. The demo shows off how to run prone and how to use it to create a coreset and use it with scikit-learn.

## Usage

After installing, import the `prone` function from the `prone` module (See `demo.py`).  It can then be used for k-means clustering. The strength of our algorithm is that it produces an approximate solution with provable guarantees much faster than previous algorithms for seeding k-means. This makes it particularly suitable to be used in conjunction with sensitivity sampling to create a coreset. This approach allows us to significantly "compress" the input data in the first stage of a clustering pipeline, which speeds up downstream tasks.

You can also import the `coreset` function to compute a coreset of your dataset. The parameters are the dataset, the number of clusters and the size of the coreset. It returns two arrays, one containing the indices of coreset points in the dataset, and the other containing the weights corresponding to the coreset points.

## Future work

Parallelism and performance improvements.

## Citing 

If you use this library in your own research, please cite our NeurIPS paper (currently this is the preprint, as the proceedings are not out yet January 5th, 2024): 
```
@inproceedings{charikar2023simple,
  title={Simple, Scalable and Effective Clustering via One-Dimensional Projections},
  author={Charikar, Moses and Henzinger, Monika and Hu, Lunjia and V{\"o}tsch, Maximilian and Waingarten, Erik},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```


