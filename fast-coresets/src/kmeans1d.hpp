#ifndef KMEANS1D_H_
#define KMEANS1D_H_

#include "utility.hpp"

namespace FastKMeans {

using projected_dataset_t = blaze::DynamicVector<coord_t>;

std::pair<std::vector<size_t>, std::vector<size_t>> efficient_k_means(projected_dataset_t &data, size_t num_centers);
std::vector<size_t> efficient_k_means(dataset_t const &data, size_t num_centers);

// the version of the 1d k-means that returns the cluster centers, point's assignments to centers and their distances
std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<coord_t>> efficient_k_means_full(dataset_t const &data,
  size_t num_centers);
std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<coord_t>>
  efficient_k_means_full_biased(dataset_t const &data, size_t num_centers);
std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<coord_t>>
  efficient_k_means_full_covariance(dataset_t const &data, size_t num_centers);
};// namespace FastKMeans


#endif// KMEANS1D_H_
