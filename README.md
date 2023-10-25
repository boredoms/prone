* `PRONE` (PRojected ONE-dimensional clustering)

This repository contains the code for the `PRONE` paper. It is currently undergoing rewrites.

A working implementation of the algorithm, which is what was used for all the experiments in the paper is can be found in the `fast-coresets` directory. It depends on `CMake`, `conan` and the `blaze` library being installed on the system. Unfortunately, conan 2.x.x is not compatible with the setup of CMake that is being used, so it might be tricky to build the submitted code.

** Future work

This repository will contain a cleaned up implementation of our algorithm as a Python pacakge to facilitate its use in the wider data science community.

Additionally, it will contain a command line utility that can be used as a stand-alone tool to cluster datasets.

** Citing 

TBD
