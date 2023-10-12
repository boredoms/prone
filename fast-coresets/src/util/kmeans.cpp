#include "kmeans.hpp"

#include <numeric>
#include <random>

#include <spdlog/spdlog.h>

namespace KMeans {

// This function computes a clustering cost of the nearest centers
coord_t clustering_cost(dataset_t const &data, std::vector<size_t> const &centers)
{
  coord_t cost{ 0.0 };

  for (size_t i = 0; i < data.rows(); i++) {
    auto point{ blaze::row(data, i) };
    coord_t min_dist{ std::numeric_limits<coord_t>::max() };

    for (auto center : centers) {
      coord_t const candidate_dist{ dist(point, blaze::row(data, center)) };

      min_dist = std::min(min_dist, candidate_dist);
    }

    cost = cost + min_dist;
  }
  return cost;
}

std::vector<coord_t> dists(dataset_t const &data, dataset_t const &centroids)
{
  std::vector<coord_t> dists(data.rows(), std::numeric_limits<coord_t>::max());

  for (size_t i = 0; i < data.rows(); i++) {
    auto point{ blaze::row(data, i) };

    for (size_t j{ 0 }; j < centroids.rows(); j++) {
      auto centroid{ blaze::row(centroids, j) };
      coord_t const candidate_dist{ dist(point, centroid) };

      dists[i] = std::min(dists[i], candidate_dist);
    }
  }
  return dists;
}

// compute clustering costs given a set of centroids
coord_t clustering_cost(dataset_t const &data, dataset_t const &centroids)
{
  coord_t cost{ 0.0 };

  for (size_t i = 0; i < data.rows(); i++) {
    auto point{ blaze::row(data, i) };
    coord_t min_dist{ std::numeric_limits<coord_t>::max() };

    for (size_t j{ 0 }; j < centroids.rows(); j++) {
      auto centroid{ blaze::row(centroids, j) };
      coord_t const candidate_dist{ dist(point, centroid) };

      min_dist = std::min(min_dist, candidate_dist);
    }

    cost += min_dist;
  }

  return cost;
}

// Choose k uniformly random centers, mainly used to provide a comparison for benchmarking, will do badly
// on datasets with a cluster structure and non-uniformly sized points
std::vector<size_t> random_centers(dataset_t const &data, size_t num_centers)
{
  auto const num_points{ data.rows() };

  std::vector<size_t> centers;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<size_t> distribution(0, num_points - 1);

  for (size_t i{ 0 }; i < num_centers; i++) { centers.emplace_back(distribution(gen)); }

  return centers;
}
}// namespace KMeans
