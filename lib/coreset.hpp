#ifndef CORESET_H_
#define CORESET_H_

#include <algorithm>
#include <random>
#include <vector>

#include "linalg.hpp"

struct coreset {
  std::vector<int> points;
  std::vector<double> weights;

  coreset(std::vector<int> points, std::vector<double> weights)
      : points(std::move(points)), weights(std::move(weights)) {}
};

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
}

#endif // CORESET_H_
