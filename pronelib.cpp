#include "pronelib.hpp"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <vector>

class sampler {
 public:
  sampler(double *data, int n, int center)
      : m_tree(2 * next_power(n) - 1, -1), n(n), leaf_start(next_power(n) - 1) {
    auto cv = data[center];

    // initialize the distribution
    for (int i = 0; i < n; i++) {
      auto d = data[i] - cv;
      m_tree[leaf_start + i] = d * d;
    }

    update_subtree(0, n - 1);
  };

  int find(double x) const {
    /**
     * If all intervals are zero we can't perform a lookup.
     *
     * I don't think a soft comparison makes that much sense here, but maybe I'm
     * wrong?
     */
    if (sum() == 0) {
      throw 0;
    }

    // Can't look up something outside of the range of intervals.
    if (x > sum()) {
      throw 1;
    }

    int i = 0;

    while (true) {
      if (i >= leaf_start) {
        return i - leaf_start;
      }

      // if it is smaller, go left
      if (x < m_tree[childl(i)]) {
        i = childl(i);
      } else {  // go right, take away left child
        x -= m_tree[childl(i)];
        i = childr(i);
      }
    }
  }

  double sum() const { return m_tree.front(); }

  /**
   * Returns the pair of upper and lower values, i.e. those of the distribution
   * that have been updated.
   */
  std::pair<int, int> update(int idx, double *data) {
    // update base layer to find upper and lower bound values
    int upper = idx + 1, lower = idx - 1;

    m_tree[leaf_start + idx] = 0;

    // first decrement lower
    while (true) {
      auto candidate = data[lower] - data[idx];
      candidate = candidate * candidate;

      if (candidate < m_tree[leaf_start + lower] && lower >= 0) {
        m_tree[leaf_start + lower] = candidate;
        lower--;
      } else {
        // done updating
        lower++;
        break;
      }
    }

    // increment upper
    while (true) {
      auto candidate = data[upper] - data[idx];
      candidate = candidate * candidate;

      if (candidate < m_tree[leaf_start + upper] && upper < n) {
        m_tree[leaf_start + upper] = candidate;
        upper++;
      } else {
        upper--;
        break;
      }
    }

    update_subtree(lower, upper);

    return std::make_pair(lower, upper);
  }

  int size() const { return m_tree.size(); };
  int num_points() const { return n; }

  /**
   * Function for debugging to print the current D^2 values of the leaves.
   */
  void print_d_squared() const {
    printf("D2 distribution: \n");
    for (int i = leaf_start; i < leaf_start + n; i++) {
      printf("%f ", m_tree[i]);
    }
    printf("\n");
  }

 private:
  inline static int childl(int i) { return 2 * i + 1; }
  inline static int childr(int i) { return 2 * i + 2; }
  inline static int parent(int i) { return (i - 1) / 2; }

  inline static int next_power(int i) {
    if (i == 1 || i == 0) {
      return i;
    }

    i = i - 1;
    int lg = log2(i);
    return 1 << (lg + 1);
  }

  /**
   * Update the subtree containing the range of leaves indicated by lower to
   * upper, including the boundary elements. The leaves are not updated. Use
   * this after making changes to the leaves to fix the nodes in the tree.
   *
   * The index of leaves is the index in the canonical array, not the internal
   * tree m_tree of this data structure.
   */
  void update_subtree(int lower_leaf_idx, int upper_leaf_idx) {
    auto lower = parent(lower_leaf_idx + leaf_start);
    auto upper = parent(upper_leaf_idx + leaf_start);

    // now update upper layers
    while (lower != upper || lower > 0) {
      update_layer(lower, upper);

      lower = parent(lower);
      upper = parent(upper);
    }

    update_layer(0, 0);
  }

  /**
   * Update a layer in the tree indicated by the range lower to upper, including
   * the lower and upper elements.
   */
  void update_layer(int lower, int upper) {
    for (int i = lower; i < upper; i++) {
      m_tree[i] = m_tree[childl(i)] + m_tree[childr(i)];
    }

    if (m_tree[childr(upper)] == -1) {
      m_tree[upper] = m_tree[childl(upper)];
    } else {
      m_tree[upper] = m_tree[childl(upper)] + m_tree[childr(upper)];
    }
  }

  std::vector<double> m_tree;
  int n;
  int leaf_start;
};

ProneKernel::ProneKernel() : random_source(std::random_device()()) {}

void ProneKernel::run(double *projected_data, int n, int k, int *centers,
                      int *assignments) {
  // first we need to sort the input data, but we need to store the index
  // permutation

  // idx[a] holds the original position of the element stored at a in the new
  // array
  std::vector<int> idx(n);
  std::iota(idx.begin(), idx.end(), 0);

  // now sort both arrays
  // sadly there seems to be no nice way to sort both arrays at once
  // TODO: see if (and if so, when) these might benefit from parallel execution
  std::sort(idx.begin(), idx.end(), [&projected_data](int i1, int i2) {
    return projected_data[i1] < projected_data[i2];
  });

  std::sort(projected_data, projected_data + n);

  // next we sample the centers
  std::vector<int> perm_assignments(n, 0);

  std::uniform_int_distribution<int> rd(0, n - 1);
  auto center = rd(random_source);

  centers[0] = center;

  // initialize the sampler, sample centers and update the distribution
  sampler sampler(projected_data, n, center);

  for (int num_sampled = 1; num_sampled < k; num_sampled++) {
    std::uniform_real_distribution<double> dist(0.0, sampler.sum());
    center = sampler.find(dist(random_source));

    centers[num_sampled] = center;

    auto [lower, upper] = sampler.update(center, projected_data);

    std::fill(perm_assignments.begin() + lower,
              perm_assignments.begin() + upper + 1, num_sampled);
  }

  // undo the permutation for centers and assignments, need to find the id of
  // the points in the original data
  for (int i = 0; i < k; i++) {
    centers[i] = idx[centers[i]];
  }

  // maybe there's a way to do this inplace?
  for (int i = 0; i < n; i++) {
    assignments[idx[i]] = perm_assignments[i];
  }
}

double distance(double *dataset, int d, int x, int y) {
  double distance = 0;

  for (int i = 0; i < d; i++) {
    double difference = dataset[d * x + i] - dataset[d * y + i];

    distance += difference * difference;
  }

  return distance;
}

void ProneKernel::coreset(double *dataset, int n, int d, int *centers, int k,
                          int *assignments, int coreset_size,
                          int *coreset_indices, double *coreset_weights) {
  // compute the distances / average costs
  std::vector<double> distances(n, 0);
  std::vector<int> cluster_sizes(k, 0);
  std::vector<double> cluster_costs(k, 0);

  double average_cost = 0.0;

  for (int i = 0; i < n; i++) {
    auto center_id = assignments[i];

    distances[i] = distance(dataset, d, i, center_id);

    average_cost += distances[i];

    cluster_sizes[center_id]++;

    cluster_costs[center_id] += distances[i];
  }

  double const ALPHA = 16 * (std::log(k) + 2);
  average_cost = average_cost / n;
  double leading_coefficient = ALPHA / average_cost;

  std::vector<double> sensitivity_precomp(k, 0);

  for (int i = 0; i < k; i++) {
    sensitivity_precomp[i] =
        (2 * ALPHA * cluster_costs[i]) / (average_cost * cluster_sizes[i]) +
        (4.0 * n) / cluster_sizes[i];
  }

  std::vector<double> sensitivity(n, 0);

  for (int i = 0; i < n; i++) {
    sensitivity[i] = leading_coefficient * distances[i] +
                     sensitivity_precomp[assignments[i]];
  }

  double total_sensitivity =
      std::reduce(sensitivity.begin(), sensitivity.end());

  std::discrete_distribution<> distribution(sensitivity.begin(),
                                            sensitivity.end());

  for (auto i = 0; i < coreset_size; i++) {
    coreset_indices[i] = distribution(random_source);
    coreset_weights[i] =
        total_sensitivity / (sensitivity[coreset_indices[i]] * coreset_size);
  }
}