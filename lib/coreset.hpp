#ifndef CORESET_H_
#define CORESET_H_

#include <algorithm>
#include <random>
#include <vector>

#include "kmeanspp.hpp"
#include "linalg.hpp"

struct coreset {
  std::vector<int> points;
  std::vector<double> weights;

  coreset(std::vector<int> points, std::vector<double> weights)
      : points(std::move(points)), weights(std::move(weights)) {}
};

/**
 * Use the lightweight coreset sampling distribution to compute a coreset.
 */
coreset lightweight_coreset(int m, int n, int d, double const *X,
                            std::mt19937 *random_source) {

  auto means = column_means(n, d, X);

  std::vector<double> sensitivity(n);
  double total = 0.0;

  for (int i = 0; i < n; i++) {
    auto d = squared_euclidean_distance(means.begin(), means.end(), X[i * d]);
    sensitivity[i] = d;
    total += d;
  }

  double offset = 1.0 / static_cast<double>(2 * n);
  total *= 2;

  std::transform(sensitivity.begin(), sensitivity.end(), sensitivity.begin(),
                 [&offset, &total](double s) { return offset + s / total; });

  std::discrete_distribution<int> distribution(sensitivity.begin(),
                                               sensitivity.end());

  std::vector<int> points(m);
  std::vector<double> weights(m);

  for (int i = 0; i < m; i++) {
    points[i] = distribution(*random_source);
    weights[i] = 1.0 / (m * sensitivity[points[i]]);
  }

  return coreset(std::move(points), std::move(weights));
}

/**
 * Given a clustering, compute a coreset using sensitivity sampling.
 */
coreset sensitivity_sampling(int m, int k, int n, int d, double const *X,
                             k_means_result const &clustering,
                             std::mt19937 *random_source) {

  std::vector<int> cluster_sizes(k, 0);
  std::vector<double> sensitivity(n), cluster_costs(k, 0.0), precomputation(k);

  double alpha = 16 * (std::log(k) + 2);
  double clustering_cost = std::reduce(clustering.distances.begin(),
                                       clustering.distances.end(), 0.0);
  double avg_cost = clustering_cost / n;
  double leading_coefficient = alpha / avg_cost;

  for (int i = 0; i < n; i++) {
    auto cluster_index = clustering.assignments[i];
    cluster_sizes[cluster_index]++;
    cluster_costs[cluster_index] += clustering.distances[i];
  }

  for (int i = 0; i < k; i++) {
    precomputation[i] =
        2 * alpha * cluster_costs[i] / (cluster_sizes[i] * avg_cost) +
        (4.0 * n) / cluster_sizes[i];
  }

  for (int i = 0; i < n; i++) {
    sensitivity[i] = leading_coefficient * clustering.distances[i] +
                     precomputation[clustering.assignments[i]];
  }

  double total_sensitivity =
      std::reduce(sensitivity.begin(), sensitivity.end(), 0.0);

  std::discrete_distribution<int> distribution(sensitivity.begin(),
                                               sensitivity.end());

  std::vector<int> points(m);
  std::vector<double> weights(m);

  for (int i = 0; i < m; i++) {
    points[i] = distribution(*random_source);
    weights[i] = total_sensitivity / (m * sensitivity[points[i]]);
  }

  return coreset(std::move(points), std::move(weights));
}

#endif // CORESET_H_
