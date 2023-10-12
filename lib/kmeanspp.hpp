#ifndef KMEANSPP_H_
#define KMEANSPP_H_

#include <random>
#include <vector>

#include "linalg.hpp"

struct k_means_result {
  std::vector<int> centers;
  std::vector<double> distances;
  std::vector<int> assignments;

  k_means_result(std::vector<int> centers, std::vector<double> distances,
                 std::vector<int> assignments)
      : centers(std::move(centers)), distances(std::move(distances)),
        assignments(std::move(assignments)) {}
};

/**
 * Run k-means++ on the dataset.
 *
 * Returns a k-means result.
 */
k_means_result kmeanspp(int k, int n, int d, double const *X,
                        std::mt19937 *random_source) {

  std::vector<double> d_square(n);
  std::vector<int> centers(k);
  std::vector<int> assignments(n, 0);

  centers[0] = std::uniform_int_distribution<int>(0, n - 1)(*random_source);

  for (int i = 0; i < n; i++) {
    d_square[i] = squared_euclidean_distance(d, i, centers[0], X);
  }

  for (int i = 1; i < k; i++) {
    std::discrete_distribution<int> d_square_distribution(d_square.begin(),
                                                          d_square.end());
    centers[i] = d_square_distribution(*random_source);

    for (int j = 0; j < n; j++) {
      auto distance = squared_euclidean_distance(d, j, centers[i], X);
      if (distance < d_square[j]) {
        d_square[j] = distance;
        assignments[j] = i;
      }
    }
  }

  return k_means_result(std::move(centers), std::move(d_square),
                        std::move(assignments));
}

/**
 * Run weighted k-means++ on the dataset.
 *
 * This is used when dealing with coresets, which are weighted datasets.
 *
 * Returns a k-means result.
 */
k_means_result weighted_kmeanspp(int k, int n, int d, double const *X,
                                 double const *W, std::mt19937 *random_source) {

  std::vector<double> d_square(n);
  std::vector<int> centers(k);
  std::vector<int> assignments(n, 0);

  centers[0] = std::uniform_int_distribution<int>(0, n - 1)(*random_source);

  for (int i = 0; i < n; i++) {
    d_square[i] = W[i] * squared_euclidean_distance(d, i, centers[0], X);
  }

  for (int i = 1; i < k; i++) {
    std::discrete_distribution<int> d_square_distribution(d_square.begin(),
                                                          d_square.end());
    centers[i] = d_square_distribution(*random_source);

    for (int j = 0; j < n; j++) {
      auto distance = W[i] * squared_euclidean_distance(d, j, centers[i], X);
      if (distance < d_square[j]) {
        d_square[j] = distance;
        assignments[j] = i;
      }
    }
  }

  return k_means_result(std::move(centers), std::move(d_square),
                        std::move(assignments));
}

#endif // KMEANSPP_H_
