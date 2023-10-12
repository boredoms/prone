#ifndef LINALG_H_
#define LINALG_H_

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

#endif // LINALG_H_
