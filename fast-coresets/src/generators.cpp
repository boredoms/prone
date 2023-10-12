#include "generators.hpp"

#include <random>

#include <spdlog/spdlog.h>

#include "utility.hpp"

namespace Generator {
// file containing utilities to generate datasets

// generate a number of clusters at distance scale from the origin, plus a small cluster at the origin
// the small cluster is likely not picked up by the LW coreset places a cluster (gaussian normal)
dataset_t symmetric_dataset(size_t inner_points, size_t outer_points, size_t num_dimensions, coord_t scale)
{
  // generate empty dataset and initialize the random number generation
  dataset_t data(inner_points + 2 * num_dimensions * outer_points, num_dimensions);

  std::random_device random_device;
  std::mt19937 rand(random_device());

  // generate a set of points around the origin
  constexpr coord_t center_radius{ 0.00 };
  constexpr coord_t cluster_radius{ 1.0 };
  constexpr coord_t cluster_distance{ 1.0 };

  std::normal_distribution<coord_t> inner_normal(0.0, center_radius);
  std::normal_distribution<coord_t> outer_normal(0.0, cluster_radius);

  size_t total_points{ 0 };

  for (size_t i{ 0 }; i < inner_points; i++) {
    auto row{ blaze::row(data, i) };

    for (auto it{ row.begin() }; it != row.end(); it++) { *it = inner_normal(rand); }

    total_points++;
  }

  // we place one cluster along each axis, with the center at radius * scale distance from the origin
  for (size_t dim{ 0 }; dim < num_dimensions; dim++) {
    auto current_total{ total_points };
    for (size_t i{ total_points }; i < current_total + 2 * outer_points; i = i + 2) {
      auto point{ blaze::row(data, i) };
      auto mirror_point{ blaze::row(data, i + 1) };
      size_t curr_dim{ 0 };

      auto mirror_it{ mirror_point.begin() };
      for (auto it{ point.begin() }; it != point.end(); it++) {
        auto value{ outer_normal(rand) };
        *it = value;
        *mirror_it = -1.0 * value;

        if (curr_dim == dim) {
          *it += cluster_distance * scale;
          *mirror_it -= cluster_distance * scale;
        }

        curr_dim++;
        mirror_it++;
      }

      total_points += 2;
      if (total_points > data.rows()) {
        spdlog::error("Number of points exceeded dataset size. This should not happen.");
        throw std::out_of_range("UwU. You wan out of spwace!!!");
      }
    }
  }

  return data;
}
}// namespace Generator
