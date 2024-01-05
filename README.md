# `PRONE` (PRojected ONE-dimensional clustering)

This repository contains a ready to use implementation of the `PRONE` algorithm from the paper "Simple, Scalable and Effective Clustering via One-Dimensional Projections".

It can be installed and used in python using `pip`. The dependencies are `numpy` and `cython`. To run the tests for the `C++` part, you need to have `CMake` installed. 

Note that this reference implementation is not what was used in the experiments of the paper, the code for this can be fonud in an older commit here.

## Usage

After installing, import the `prone` function from the `prone` module. It can then be used for k-means clustering. The strength of our algorithm is that it produces an approximate solution with provable guarantees much faster than previous algorithms for seeding k-means. This makes it particularly suitable to be used in conjunction with sensitivity sampling to create a coreset. This approach allows us to significantly "compress" the input data in the first stage of a clustering pipeline, which speeds up downstream tasks.

## Future work

In the next version, we will add a reference coreset construction.

## Citing 

If you use this library in your own research, please cite our NeurIPS paper (currently this is the preprint, as the proceedings are not out yet January 5th, 2024): 
```@article{charikar2023simple,
  title={Simple, Scalable and Effective Clustering via One-Dimensional Projections},
  author={Charikar, Moses and Henzinger, Monika and Hu, Lunjia and V{\"o}tsch, Maxmilian and Waingarten, Erik},
  journal={arXiv preprint arXiv:2310.16752},
  year={2023}
}
```
