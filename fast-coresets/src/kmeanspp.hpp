#ifndef KMEANSPP_H_
#define KMEANSPP_H_
#pragma once

#include "utility.hpp"

namespace KMeansPlusPlus {
// local search heuristic for improving k-means++, after O(k) (large constant) additional steps
// this will yield a constant factor approximation
// TODO port this back to blaze (low priority)
// std::vector<size_t> local_search(dataset_t const &data, std::vector<size_t> &centers, size_t num_steps);
// classic k-means++ of Arthur and Vassilvitski, returning only the computed centers
// This is just a wrapper around run_full that discards the second and third elements of the
// returned tuple.
std::vector<size_t> run(dataset_t const &data, size_t num_centers);
// classic k-means++, returns centers, assignments and distances
std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<coord_t>> run_full(dataset_t const &data,
  size_t num_centers);
}// namespace KMeansPlusPlus

#endif// KMEANSPP_H_
