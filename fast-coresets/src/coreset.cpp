#include "coreset.hpp"

#include <algorithm>
#include <numeric>
#include <random>

#include <spdlog/spdlog.h>

#include "kmeans1d.hpp"
#include "kmeanspp.hpp"

namespace SensitivitySampling {

// the algorithm from Bachem, Lucic, Krause's survey
coreset_t run(dataset_t const &data,
  centers_t const &centers,
  std::vector<size_t> const &assignments,
  std::vector<coord_t> const &distances,
  std::vector<size_t> const &cluster_sizes,
  std::vector<coord_t> const &cluster_costs,
  size_t coreset_size)
{
  coord_t const ALPHA = 16 * (std::log(centers.size()) + 2);
  coord_t const num_points{ static_cast<coord_t>(data.rows()) };
  coord_t const avg_cost{ std::reduce(cluster_costs.begin(), cluster_costs.end()) / num_points };
  coord_t const leading_coefficient{ ALPHA / avg_cost };

  std::vector<coord_t> sensitivity(data.rows());

  std::vector<coord_t> sensitivity_precomputation(centers.size());

  for (size_t i{ 0 }; i < centers.size(); i++) {
    coord_t cluster_size{ static_cast<coord_t>(cluster_sizes[i]) };
    sensitivity_precomputation[i] =
      (2.0 * ALPHA * cluster_costs[i]) / (cluster_size * avg_cost) + (4.0 * num_points) / cluster_size;
  }

  // omp parallel for did not seem to speed this up for 100k size data
  // #pragma omp parallel for
  for (size_t i = 0; i < data.rows(); i++) {
    sensitivity[i] = leading_coefficient * distances[i] + sensitivity_precomputation[assignments[i]];
  }
  coord_t const total_sensitivity = std::reduce(sensitivity.cbegin(), sensitivity.cend());

  std::random_device rd;
  std::mt19937 rand{ rd() };

  std::discrete_distribution<size_t> distribution(sensitivity.begin(), sensitivity.end());

  centers_t indices;
  std::vector<coord_t> weights;

  for (size_t i{ 0 }; i < coreset_size; i++) {
    auto sample = distribution(rand);
    indices.emplace_back(sample);
    weights.emplace_back(total_sensitivity / (static_cast<coord_t>(coreset_size) * sensitivity[sample]));
  }

  return { indices, weights };
}

coreset_t run(dataset_t const &data,
  centers_t const &centers,
  std::vector<size_t> const &assignments,
  std::vector<coord_t> const &distances,
  std::vector<size_t> const &cluster_sizes,
  size_t coreset_size)
{
  // compute cluster costs
  std::vector<coord_t> cluster_costs(centers.size(), 0.0);

  // TODO parallelize
  for (size_t i{ 0 }; i < data.rows(); i++) { cluster_costs[assignments[i]] += distances[i]; }

  return run(data, centers, assignments, distances, cluster_sizes, cluster_costs, coreset_size);
}

coreset_t run(dataset_t const &data,
  centers_t const &centers,
  std::vector<size_t> const &assignments,
  std::vector<size_t> const &cluster_sizes,
  size_t coreset_size)
{
  // compute distances
  std::vector<coord_t> distances(data.rows());

  // #pragma omp parallel for
  for (size_t i = 0; i < data.rows(); i++) {
    auto closest_center = centers[assignments[i]];
    distances[i] = KMeans::dist(blaze::row(data, i), blaze::row(data, closest_center));
  }

  return run(data, centers, assignments, distances, cluster_sizes, coreset_size);
}

coreset_t run(dataset_t const &data,
  centers_t const &centers,
  std::vector<size_t> const &assignments,
  std::vector<double> const &distances,
  size_t coreset_size)
{
  // compute cluster sizes
  std::vector<size_t> cluster_sizes(centers.size(), 0);
  // TODO parallelize
  for (auto assignment : assignments) { cluster_sizes[assignment]++; }

  return run(data, centers, assignments, distances, cluster_sizes, coreset_size);
}

coreset_t
  run(dataset_t const &data, centers_t const &centers, std::vector<size_t> const &assignments, size_t coreset_size)
{
  // compute cluster sizes
  size_t num_centers{ centers.size() };
  std::vector<size_t> cluster_sizes(num_centers, 0);
  // TODO parallelize
  for (auto cluster : assignments) { cluster_sizes[cluster]++; }

  return run(data, centers, assignments, cluster_sizes, coreset_size);
}

// compute a coreset from a set of centers, this is rather expensive as we need to do n * k * d operations,
// try and prevent recomputing as much as possible, k = size of centers
coreset_t run(dataset_t const &data, centers_t const &centers, size_t coreset_size)
{
  std::vector<size_t> assignments(data.rows());
  std::vector<coord_t> distances(data.rows(), std::numeric_limits<coord_t>::max());

  // #pragma omp parallel for collapse(2)
  for (size_t i = 0; i < centers.size(); i++) {
    for (size_t j = 0; j < data.rows(); j++) {
      auto distance = KMeans::dist(blaze::row(data, j), blaze::row(data, centers[j]));
      if (distance < distances[j]) {
        distances[j] = distance;
        assignments[j] = i;
      }
    }
  }

  return run(data, centers, assignments, distances, coreset_size);
}

struct kmeans_result_t
{
  coord_t cost;
  std::vector<size_t> centers;
  std::vector<size_t> assignments;
  std::vector<coord_t> distances;

  kmeans_result_t(std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<coord_t>> &&result)
  {
    centers = std::get<0>(result);
    assignments = std::get<1>(result);
    distances = std::get<2>(result);
    cost = std::reduce(distances.cbegin(), distances.cend());
  }

  kmeans_result_t() { cost = std::numeric_limits<coord_t>::max(); }
};

// compute a sensitivity sampling coreset using kmeans++ as the bicriteria
coreset_t run(dataset_t const &data, size_t num_centers, size_t coreset_size, double delta)
{
  spdlog::info("Computing the approximate solution using k-means++");
  size_t const n_tries = static_cast<size_t>(std::log(1.0 / (1.0 - delta)));
  spdlog::info("Number of tries for delta={} is {}", delta, n_tries);

  kmeans_result_t res{ KMeansPlusPlus::run_full(data, num_centers) };

  for (size_t i = 1; i < n_tries; i++) {
    kmeans_result_t new_res{ KMeansPlusPlus::run_full(data, num_centers) };

    if (new_res.cost < res.cost) { res = new_res; }
  }

  return run(data, res.centers, res.assignments, res.distances, coreset_size);
}

// use our algorithm instead of k-means++
coreset_t run_near_linear(dataset_t const &data, size_t num_centers, size_t coreset_size, double delta)
{
  spdlog::info("Computing the approximate solution using fast k-means++");
  size_t const n_tries = static_cast<size_t>(std::log(1.0 / (1.0 - delta)));
  spdlog::info("Number of tries for delta={} is {}", delta, n_tries);

  kmeans_result_t res;

  for (size_t i = 0; i < n_tries; i++) {
    kmeans_result_t new_res{ FastKMeans::efficient_k_means_full(data, num_centers) };

    if (new_res.cost < res.cost) { res = new_res; }
  }

  return run(data, res.centers, res.assignments, res.distances, coreset_size);
}

coreset_t run_near_linear_biased(dataset_t const &data, size_t num_centers, size_t coreset_size, double delta)
{
  spdlog::info("Computing the approximate solution using fast k-means++");
  size_t const n_tries = static_cast<size_t>(std::log(1.0 / (1.0 - delta)));
  spdlog::info("Number of tries for delta={} is {}", delta, n_tries);

  kmeans_result_t res;

  for (size_t i = 0; i < n_tries; i++) {
    kmeans_result_t new_res{ FastKMeans::efficient_k_means_full_biased(data, num_centers) };

    if (new_res.cost < res.cost) { res = new_res; }
  }

  return run(data, res.centers, res.assignments, res.distances, coreset_size);
}

coreset_t run_near_linear_covariance(dataset_t const &data, size_t num_centers, size_t coreset_size, double delta)
{
  spdlog::info("Computing the approximate solution using fast k-means++");
  size_t const n_tries = static_cast<size_t>(std::log(1.0 / (1.0 - delta)));
  spdlog::info("Number of tries for delta={} is {}", delta, n_tries);

  kmeans_result_t res;

  for (size_t i = 0; i < n_tries; i++) {
    kmeans_result_t new_res{ FastKMeans::efficient_k_means_full_covariance(data, num_centers) };

    if (new_res.cost < res.cost) { res = new_res; }
  }

  return run(data, res.centers, res.assignments, res.distances, coreset_size);
}

}// namespace SensitivitySampling
