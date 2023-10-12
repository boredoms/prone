#ifndef GENERATE_DATASET_H_
#define GENERATE_DATASET_H_

#include <random>

/**
 * Function to generate the Gaussian dataset as used in the experiments.
 *
 * Returns a dataset of size (inner_points + 2 * d * outer_points, d)
 */
double *generate_gaussian_dataset(int d, int inner_points, int outer_points,
                                  double cluster_distance,
                                  double inner_radius = 0.0,
                                  double outer_radius = 1.0) {

  int num_points = inner_points + 2 * d * outer_points;

  double *X = new double[num_points * d];

  std::random_device random_device;
  std::mt19937 random_source(random_device());

  std::normal_distribution<double> inner_normal(0.0, inner_radius);
  std::normal_distribution<double> outer_normal(0.0, outer_radius);

  for (int i = 0; i < inner_points * d; i++) {
    X[i] = inner_normal(random_source);
  }

  // initialize the outer clusters
  for (int i = inner_points * d; i < (inner_points + outer_points * d) * d;
       i++) {
    X[i] = outer_normal(random_source);
  }

  // shift the outer clusters
  for (int i = 0; i < d; i++) {
    for (int j = inner_points + i * outer_points;
         j < inner_points + (i + 1) * outer_points; j++) {
      X[j * d + i] += cluster_distance;
    }
  }

  // mirror
  int offset = outer_points * d * d;

  for (int i = inner_points * d; i < (inner_points + outer_points * d) * d;
       i++) {
    X[i + offset] = -1.0 * X[i];
  }

  return X;
}

#endif // GENERATE_DATASET_H_
