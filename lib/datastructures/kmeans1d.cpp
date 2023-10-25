#include "kmeans1d.hpp"

#include <chrono>
#include <execution>
#include <iostream>
#include <numeric>
#include <random>

#include <spdlog/spdlog.h>

using projected_dataset_t = blaze::DynamicVector<coord_t>;

namespace FastKMeans {
// this needs the type to have a random access iterator and support random access through the [] operator
template<typename T> std::vector<size_t> sort_indices(T const &list)// NOLINT(readibility-identifier-length)
{
  // create a vector containing the indices
  std::vector<size_t> indices(list.size());
  std::iota(indices.begin(), indices.end(), 0);

  // sort the indices based on their value
  std::sort(// std::execution::par_unseq,
    indices.begin(),
    indices.end(),
    [&list](size_t first, size_t second) { return list[first] < list[second]; });

  return indices;
}

projected_dataset_t map_to_1D(dataset_t const &data);

// TODO the logic of this class is brittle, make it better
class dynamic_tree
{
  inline static size_t left_child(size_t index) { return 2 * index + 1; }
  inline static size_t right_child(size_t index) { return 2 * index + 2; }
  inline static size_t parent(size_t index) { return (index - 1) / 2; }

public:
  explicit dynamic_tree(std::vector<coord_t> const &distances)
    : m_size(distances.size() - 1), m_dists(&distances), m_tree(distances.size() - 1)
  {
    if (m_size > 0) {
      for (size_t i{ m_size - 1 }; i > 0; i--) {
        if (right_child(i) < m_size) {// internal node, no leaves
          m_tree[i] = m_tree[left_child(i)] + m_tree[right_child(i)];
        } else if (left_child(i) < m_size) {
          m_tree[i] = m_tree[left_child(i)] + distances[right_child(i) - m_size];
        } else {
          m_tree[i] = distances[left_child(i) - m_size] + distances[right_child(i) - m_size];
        }
      }

      // finally fix the root
      if (2 < m_size) {// internal node, no leaves
        m_tree[0] = m_tree[1] + m_tree[2];
      } else if (1 < m_size) {
        m_tree[0] = m_tree[1] + distances[2 - m_size];
      } else {
        m_tree[0] = distances[1 - m_size] + distances[2 - m_size];
      }
    } else {
      spdlog::warn("The tree is empty.");
    }
  }

  void update(std::vector<coord_t> const &distances, size_t lower, size_t upper)
  {
    // calculate the parent range, update, recurse
    upper += m_size;
    lower += m_size;

    do {
      upper = parent(upper);
      lower = parent(lower);

      for (size_t i{ upper }; i >= lower; i--) {
        m_tree[i] = 0.0;
        if (right_child(i) < m_size) {
          m_tree[i] += m_tree[right_child(i)];
        } else {
          m_tree[i] += distances[right_child(i) - m_size];
        }

        if (left_child(i) < m_size) {
          m_tree[i] += m_tree[left_child(i)];
        } else {
          m_tree[i] += distances[left_child(i) - m_size];
        }

        if (i == 0) { break; }// prevents i from underflowing
      }
    } while (lower != 0);
  }

  [[nodiscard]] size_t find(coord_t value) const// this feels needlessly complicated
  {
    size_t index{ 0 };
    while (index < m_size) {
      if (right_child(index) >= m_size) {
        if (left_child(index) >= m_size) {// both are leaves
          if (value < (*m_dists)[left_child(index) - m_size]) {
            return left_child(index) - m_size;
          } else {
            return right_child(index) - m_size;
          }
        } else {// right child is leaf
          if (value > m_tree[left_child(index)]) {// return right child or
            return right_child(index) - m_size;
          } else {
            index = left_child(index);// go to left child
          }
        }
      } else {// both children are internal nodes
        if (value < m_tree[left_child(index)]) {
          index = left_child(index);// go to the left
        } else {
          value -= m_tree[left_child(index)];// go to the right, need to subtract the left subtree weight
          index = right_child(index);
        }
      }
    }
    spdlog::error("This should never be reached.");
    return std::numeric_limits<size_t>::max();
  }
  [[nodiscard]] coord_t sum() const { return m_tree[0]; }

private:
  size_t m_size;
  std::vector<coord_t> const
    *m_dists;// this might go horribly wrong if it points to an object with a lifetime less than the datastructure
  std::vector<coord_t> m_tree;
};

// this function will reorder the data
std::pair<std::vector<size_t>, std::vector<size_t>> efficient_k_means(projected_dataset_t &data, size_t num_centers)
{
  // // data is a 1d vector of reduced points, multiplication works
  // for (auto it = data.begin(); it != data.end(); it++) { spdlog::info("The data is: {}", *it); }

  // check the preconditions on k
  if (num_centers == 0) {
    spdlog::warn("num_centers = 0, returning empty vector");
    return {};
  }

  if (blaze::isEmpty(data)) {
    spdlog::warn("Dataset is empty, can not return any centers.");
    return {};
  }

  size_t const num_points{ data.size() };

  if (num_centers >= num_points) {
    spdlog::info("(num_centers := {} >= n): More centers than points requested, returning iota", num_centers);
    std::vector<size_t> centers(num_points);
    std::iota(centers.begin(), centers.end(), 0);
    return { centers, centers };
  }

  // set up randomness for the k-means procedure
  std::random_device random_device;
  std::mt19937 rand{ random_device() };

  // sort data and get a map of indices for the points
  std::vector<size_t> const sorted_indices{ sort_indices(data) };
  std::sort(// std::execution::par_unseq,
    data.begin(),
    data.end());

  // for (auto i = 0; i < data.size(); i++) {
  //   spdlog::info("The sorted data is: {}, originally at index {}", data[i], sorted_indices[i]);
  // }

  // generate the first center
  std::uniform_int_distribution<size_t> initial_dist(0, num_points - 1);
  std::vector<size_t> centers;
  centers.emplace_back(initial_dist(rand));

  spdlog::info("First center is {} originally {}", centers[0], sorted_indices[centers[0]]);

  std::vector<size_t> assignments(num_points, 0);

  std::vector<coord_t> distances(num_points);// initialize the vector of distances
  size_t const first_center{ centers.front() };
  for (size_t i{ 0 }; i < num_points; i++) {
    distances[i] = (data[i] - data[first_center]) * (data[i] - data[first_center]);
  }

  dynamic_tree dynamic_tree(distances);

  // main for loop of the theorem
  for (size_t i{ 1 }; i < num_centers; i++) {
    // for (auto v : assignments) { spdlog::info("Assignments: {}", v); }
    // for (auto d : distances) { spdlog::info("Distances: {}", d); }

    // spdlog::info("Sum: {}", dynamic_tree.sum());

    std::uniform_real_distribution<coord_t> dist(0.0, dynamic_tree.sum());
    auto const sample{ dist(rand) };

    size_t const new_center{ dynamic_tree.find(sample) };

    centers.emplace_back(new_center);
    // spdlog::info("Next center is {} originally {}", centers[i], sorted_indices[centers[i]]);
    distances[new_center] = 0.0;
    assignments[new_center] = i;

    size_t upper{ new_center + 1 };
    int lower{ static_cast<int>(new_center) - 1 };

    coord_t candidate_distance;
    while (lower >= 0
           && ((candidate_distance = (data[lower] - data[new_center]) * (data[lower] - data[new_center]))
               < distances[lower])) {
      distances[lower] = candidate_distance;
      assignments[lower] = i;
      lower--;
    }

    while (upper < num_points
           && ((candidate_distance = (data[upper] - data[new_center]) * (data[upper] - data[new_center]))
               < distances[upper])) {
      distances[upper] = candidate_distance;
      assignments[upper] = i;
      upper++;
    }

    dynamic_tree.update(distances, lower + 1, upper - 1);
  }
  // for (auto v : assignments) { spdlog::info("Assignments: {}", v); }
  // for (auto d : distances) { spdlog::info("Distances: {}", d); }
  //  for (auto d : distances) { spdlog::info("Distances: {}", d); }

  // map the points back to their original ones
  for (size_t i{ 0 }; i < num_centers; i++) { centers[i] = sorted_indices[centers[i]]; }

  std::vector<size_t> permuted_assignments(data.size());
  for (size_t i{ 0 }; i < data.size(); i++) { permuted_assignments[sorted_indices[i]] = assignments[i]; }

  return { centers, permuted_assignments };
}

std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<coord_t>> efficient_k_means_full(dataset_t const &data,
  size_t num_centers)
{
  auto reduced_data{ map_to_1D(data) };// do JLT reduction
  auto [centers, assignments] = efficient_k_means(reduced_data, num_centers);

  std::vector<coord_t> distances(data.rows());

  // spdlog::info("Data: {}, Centers: {}, Assignments: {}, Dists: {}",
  //   data.rows(),
  //   centers.size(),
  //   assignments.size(),
  //   distances.size());
  int active_level = omp_get_active_level();
  // #pragma omp parallel for if (active_level < 1)
  for (size_t i = 0; i < data.rows(); i++) {
    auto point{ blaze::row(data, i) };
    auto center{ blaze::row(data, centers[assignments[i]]) };
    distances[i] = KMeans::dist(point, center);
  }

  // spdlog::info("Clustering cost from assigned centers is {}", std::reduce(distances.begin(), distances.end()));

  return std::make_tuple(centers, assignments, distances);
}

std::vector<size_t> efficient_k_means(dataset_t const &data, size_t num_centers)
{
  auto [centers, assignments, distances] = efficient_k_means_full(data, num_centers);
  return centers;
}

// function to compute the empirical covariance matrix
dataset_t compute_covariance_matrix(dataset_t const &data)
{
  // calculate the column means
  spdlog::info("Computing the covariance matrix");
  auto col_mean = blaze::mean<blaze::columnwise>(data);
  auto centered_data{ data };

  spdlog::info("Centering the dataset");
  for (size_t i{ 0 }; i < data.columns(); i++) {
    auto col = blaze::row(centered_data, i);
    col = col - col_mean[i];
  }

  return blaze::trans(centered_data) * centered_data * (1.0 / static_cast<coord_t>(blaze::rows(data)));
}

// sample a multivariate normal according to the empirical covariance matrix
// of the data
blaze::DynamicVector<coord_t> multivariate_normal(dataset_t const &data)
{
  // auto col_mean = blaze::mean<blaze::columnwise>(data);

  // first create a gaussian normal vector
  std::random_device random_device;
  std::mt19937 rand{ random_device() };
  std::normal_distribution<> normal;

  blaze::DynamicVector<coord_t> normal_vec(data.columns());
  for (size_t i{ 0 }; i < data.columns(); i++) { normal_vec[i] = normal(rand); }

  auto covariance_matrix{ compute_covariance_matrix(data) };

  spdlog::info(
    "Covariance matrix is of size {} by {}", blaze::rows(covariance_matrix), blaze::columns(covariance_matrix));

  dataset_t norm_transform;
  blaze::llh(covariance_matrix, norm_transform);

  normal_vec = norm_transform * normal_vec;

  // for (size_t i{ 0 }; i < normal_vec.size(); i++) { normal_vec[i] += col_mean[i]; }

  return normal_vec;
}

// this seems fine
projected_dataset_t map_to_1D(dataset_t const &data)// NOLINT(readibility-identifier-length)
{
  // check if dataset is empty
  if (blaze::isEmpty(data)) { return {}; }

  // first create a gaussian normal vector
  std::random_device random_device;
  std::mt19937 rand{ random_device() };
  std::normal_distribution<> normal;

  // create and initialize the vector of standard normal random variables
  blaze::DynamicVector<coord_t> normal_vec(data.columns());
  for (auto it{ normal_vec.begin() }; it != normal_vec.end(); it++) { *it = normal(rand); }

  // for (auto it{ normal_vec.begin() }; it != normal_vec.end(); it++) { spdlog::info("normal: {}", *it); }

  return data * normal_vec;
}

// biased mapping
projected_dataset_t map_to_1D_biased(dataset_t const &data)// NOLINT(readibility-identifier-length)
{
  // check if dataset is empty
  if (blaze::isEmpty(data)) { return {}; }

  // auto col_mean = blaze::mean<blaze::columnwise>(data);
  auto col_stddev = blaze::stddev<blaze::columnwise>(data);

  // first create a gaussian normal vector
  std::random_device random_device;
  std::mt19937 rand{ random_device() };
  std::normal_distribution<> normal;

  // create and initialize the vector of standard normal random variables
  blaze::DynamicVector<coord_t> normal_vec(data.columns());
  for (size_t i{ 0 }; i < data.columns(); i++) { normal_vec[i] = col_stddev[i] * normal(rand); }// + col_mean[i]; }

  normal_vec = blaze::normalize(normal_vec);

  return data * normal_vec;
}

// biased mapping with covariance
projected_dataset_t map_to_1D_covariance(dataset_t const &data)// NOLINT(readibility-identifier-length)
{
  // check if dataset is empty
  if (blaze::isEmpty(data)) { return {}; }

  // create and initialize the vector of standard normal random variables
  blaze::DynamicVector<coord_t> normal_vec{ multivariate_normal(data) };

  normal_vec = blaze::normalize(normal_vec);

  return data * normal_vec;
}

std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<coord_t>>
  efficient_k_means_full_biased(dataset_t const &data, size_t num_centers)
{
  auto reduced_data{ map_to_1D_biased(data) };// do JLT reduction
  auto [centers, assignments] = efficient_k_means(reduced_data, num_centers);

  std::vector<coord_t> distances(data.rows());

  // spdlog::info("Data: {}, Centers: {}, Assignments: {}, Dists: {}",
  //   data.rows(),
  //   centers.size(),
  //   assignments.size(),
  //   distances.size());

  int active_level = omp_get_active_level();
  // #pragma omp parallel for if (active_level < 1)
  for (size_t i = 0; i < data.rows(); i++) {
    auto point{ blaze::row(data, i) };
    auto center{ blaze::row(data, centers[assignments[i]]) };
    distances[i] = KMeans::dist(point, center);
  }

  // spdlog::info("Clustering cost from assigned centers is {}", std::reduce(distances.begin(), distances.end()));

  return std::make_tuple(centers, assignments, distances);
}

std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<coord_t>>
  efficient_k_means_full_covariance(dataset_t const &data, size_t num_centers)
{
  auto reduced_data{ map_to_1D_covariance(data) };// do JLT reduction
  auto [centers, assignments] = efficient_k_means(reduced_data, num_centers);

  std::vector<coord_t> distances(data.rows());

  // spdlog::info("Data: {}, Centers: {}, Assignments: {}, Dists: {}",
  //   data.rows(),
  //   centers.size(),
  //   assignments.size(),
  //   distances.size());

  int active_level = omp_get_active_level();
  // #pragma omp parallel for if (active_level < 1)
  for (size_t i = 0; i < data.rows(); i++) {
    auto point{ blaze::row(data, i) };
    auto center{ blaze::row(data, centers[assignments[i]]) };
    distances[i] = KMeans::dist(point, center);
  }

  // spdlog::info("Clustering cost from assigned centers is {}", std::reduce(distances.begin(), distances.end()));

  return std::make_tuple(centers, assignments, distances);
}

};// namespace FastKMeans
