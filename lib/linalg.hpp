#ifndef LINALG_H_
#define LINALG_H_

#include <vector>

/**
 * Calculate the squared euclidean distance of two datapoints in the dataset X
 *
 * Args:
 * @d Dimension of the dataset.
 * @i Index of the first point.
 * @j Index of the second point.
 * @X The dataset, of dimension (n, d).
 */
double squared_euclidean_distance(int d, int i, int j, double const *X) {
  double distance = 0.0;

  for (int k = 0; k < d; k++) {
    distance += std::pow(X[i * d + k] - X[j * d + k], 2);
  }

  return distance;
}

template <typename it, typename it_>
double squared_euclidean_distance(it begin, it end, it_ iter_) {
  double distance = 0.0;

  for (it iter = begin; iter != end; iter++, iter_++) {
    distance += std::pow(*iter - *iter_, 2);
  }

  return distance;
}

/**
 * Calculate the mean of each feature in the dataset.
 *
 */
std::vector<double> column_means(int n, int d, double const *X) {
  std::vector<double> means(d, 0.0);

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {
      means[j] += X[i * d + j];
    }
  }

  for (int i = 0; i < d; i++) {
    means[i] /= n;
  }

  return means;
}

#endif // LINALG_H_
