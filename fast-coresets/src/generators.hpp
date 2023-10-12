#ifndef GENERATORS_H_
#define GENERATORS_H_

#include "utility.hpp"
namespace Generator {
dataset_t symmetric_dataset(size_t inner_points, size_t outer_points, size_t num_dimensions, coord_t scale);
}

#endif// GENERATORS_H_
