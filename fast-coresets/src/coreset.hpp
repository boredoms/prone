#ifndef CORESET_H_
#define CORESET_H_

#include "utility.hpp"

namespace SensitivitySampling {
// sensitivity sampling based coreset constructions
// different functions exist to prevent doing too much work, for example if the initialization
// provides assignments / distances or other things
coreset_t run(dataset_t const &data,
  centers_t const &centers,
  std::vector<size_t> const &assignments,
  std::vector<coord_t> const &distances,
  std::vector<size_t> const &cluster_sizes,
  std::vector<coord_t> const &cluster_costs,
  size_t coreset_size);

coreset_t run(dataset_t const &data,
  centers_t const &centers,
  std::vector<size_t> const &assignments,
  std::vector<coord_t> const &distances,
  std::vector<size_t> const &cluster_sizes,
  size_t coreset_size);

coreset_t run(dataset_t const &data,
  centers_t const &centers,
  std::vector<size_t> const &assignments,
  std::vector<size_t> const &cluster_sizes,
  size_t coreset_size);

coreset_t run(dataset_t const &data,
  centers_t const &centers,
  std::vector<size_t> const &assignments,
  std::vector<double> const &distances,
  size_t coreset_size);

coreset_t run(dataset_t const &data, size_t k, centers_t const &centers, size_t m);

coreset_t run(dataset_t const &data, size_t num_centers, size_t coreset_size, double delta = 0.99);

coreset_t run_near_linear(dataset_t const &data, size_t num_centers, size_t coreset_size, double delta = 0.99);
coreset_t run_near_linear_biased(dataset_t const &data, size_t num_centers, size_t coreset_size, double delta = 0.99);
coreset_t
  run_near_linear_covariance(dataset_t const &data, size_t num_centers, size_t coreset_size, double delta = 0.99);
}// namespace SensitivitySampling

#endif// CORESET_H_
