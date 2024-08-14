#include <random>

class ProneKernel {
 public:
  ProneKernel();
  /**
   * A major component of the PRONE algorithm, solving k-means++ in one
   * dimension. See sections 5 and 6 in https://arxiv.org/pdf/2310.16752.pdf
   *
   * Parameters:
   *    projected_data: A size n array containing the projected elements.
   *    n: Number of data points.
   *    k: Number of centers to sample.
   *    centers: A size k array, to which the indices of centers will be
   * written.
   *    assignments: A size n array, to which the assignment of points
   * will be written.
   */
  void run(double *projected_data, int n, int k, int *centers,
           int *assignments);

  void coreset(double *dataset, int n, int d, int *centers, int k,
               int *assignments, int coreset_size, int *coreset_indices,
               double *coreset_weights);

 private:
  std::mt19937 random_source;
};