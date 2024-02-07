# `PRONE` (PRojected ONE-dimensional clustering)

This repository contains a ready to use implementation of the `PRONE` algorithm from the paper "Simple, Scalable and Effective Clustering via One-Dimensional Projections". This is a reference implementation published alongside our paper.

It can be installed and used in python using `pip`. The dependencies are `numpy` and `cython`. To run the tests for the C++ code, you need to have `CMake` installed. 

## Usage

After installing, import the `prone` function from the `prone` module (See `demo.py`).  It can then be used for k-means clustering. The strength of our algorithm is that it produces an approximate solution with provable guarantees much faster than previous algorithms for seeding k-means. This makes it particularly suitable to be used in conjunction with sensitivity sampling to create a coreset. This approach allows us to significantly "compress" the input data in the first stage of a clustering pipeline, which speeds up downstream tasks.

## Future work

In the next version, we will add a reference coreset construction.

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


