#ifndef KMEANS_H_
#define KMEANS_H_
#pragma once

#include <numeric>

#include "types.hpp"

namespace KMeans {

// calculate the distance between two points, O(d) time, sadly uses a new allocation, but blaze does not have a distance
// function afaik, very generic template since row types can be a matrix row and dynamic vector
template<typename row_type, typename row_type_> coord_t dist(row_type &first, row_type_ &second)
{
  return blaze::sqrNorm(first - second);
}

// functions to calculate the clustering cost of a set of centers given either as a list of indices or a dataset in the
// case of steiner points
coord_t clustering_cost(dataset_t const &data, std::vector<size_t> const &centers);
coord_t clustering_cost(dataset_t const &data, dataset_t const &centroids);

// function for calculating the closest distances of all points in data to the centroids
std::vector<coord_t> dists(dataset_t const &data, dataset_t const &centroids);

// this just picks num_centers many centers uniformly at random
std::vector<size_t> random_centers(dataset_t const &data, size_t num_centers);
}// namespace KMeans

#endif// KMEANS_H_
