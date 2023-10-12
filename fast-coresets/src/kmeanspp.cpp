#include "kmeanspp.hpp"

#include <algorithm>
#include <execution>
#include <random>
#include <utility>

#include <spdlog/spdlog.h>

namespace KMeansPlusPlus {

// // TODO uncurse this, let users pass the arguments along that have been computed already
// // medium implementation, keeps the closest and second closest distances
// std::vector<size_t> local_search(dataset_t const &data, std::vector<size_t> &centers, size_t num_steps)
// {
//   std::vector<coord_t> dists(data.size(), std::numeric_limits<coord_t>::max());
//   std::vector<coord_t> second_closest_dists(data.size(), std::numeric_limits<coord_t>::max());
//   std::vector<size_t> closest_center(data.size());// to keep track of the indices
//   std::vector<size_t> second_closest_center(data.size());

//   // initialize dists
//   for (size_t i{ 0 }; i < data.size(); i++) {
//     for (auto k : centers) {
//       auto dist = KMeans::dist(data[i], data[k]);
//       if (dist < dists[i]) {// if distance is new minimum
//         second_closest_dists[i] = dists[i];
//         second_closest_center[i] = closest_center[i];
//         dists[i] = dist;
//         closest_center[i] = k;
//       } else if (dist < second_closest_dists[i]) {
//         second_closest_dists[i] = dist;
//         second_closest_center[i] = k;
//       }
//     }
//   }
//   // dists and second closest now contain distances of each point to its two closest centers

//   auto clustering_cost{ std::reduce(dists.begin(), dists.end()) };
//   spdlog::info("Starting local search, initial cost = {}\n", clustering_cost);

//   std::random_device rd;
//   std::mt19937 gen(rd());

//   for (size_t i{ 0 }; i < num_steps; i++) {
//     std::discrete_distribution<size_t> distrib(dists.begin(), dists.end());
//     auto candidate{ distrib(gen) };// get a new candidate

//     std::vector<coord_t> candidate_dists(data.size(), std::numeric_limits<coord_t>::max());
//     // initialize candidate distances
//     for (size_t i{ 0 }; i < data.size(); i++) { candidate_dists[i] = KMeans::dist(data[i], data[candidate]); }

//     // got the coordinate dists
//     std::vector<coord_t> candidate_costs(
//       centers.size(), 0.0);// can compute all alternative clusterings in a single pass

//     for (size_t i{ 0 }; i < centers.size(); i++) {// caluclate the clustering cost for each possible replacement
//       for (size_t j{ 0 }; j < data.size(); j++) {
//         if (closest_center[j] == centers[i]) {// if the closest center is the center we are replacing
//           candidate_costs[i] += std::min(second_closest_dists[j], candidate_dists[j]);
//         } else {
//           candidate_costs[i] += std::min(dists[j], candidate_dists[j]);
//         }
//       }
//     }

//     auto it = std::min_element(candidate_costs.begin(), candidate_costs.end());
//     auto min_index{ std::distance(candidate_costs.begin(), it) };

//     if (candidate_costs[min_index] < clustering_cost) {// if the best choice is better than before
//       auto removed_center = centers[min_index];
//       centers[min_index] = candidate;// replace the center, should work fine

//       // update the distances
//       for (size_t i{ 0 }; i < data.size(); i++) {
//         // if the closest or second closest distance is the removed center, we need to recompute the whole thing
//         if (closest_center[i] == removed_center || second_closest_center[i] == removed_center) {
//           dists[i] = std::numeric_limits<coord_t>::max();
//           for (auto k : centers) {
//             auto dist = KMeans::dist(data[i], data[k]);
//             if (dist < dists[i]) {
//               second_closest_dists[i] = dists[i];
//               second_closest_center[i] = closest_center[i];

//               dists[i] = dist;
//               closest_center[i] = k;

//             } else if (dist < second_closest_dists[i]) {

//               second_closest_dists[i] = dist;
//               second_closest_center[i] = k;
//             }
//           }// this branch is okay now
//         } else {// here we just update by taking minimuns
//           auto dist = candidate_dists[i];
//           if (dist < dists[i]) {// just update the regular way
//             second_closest_dists[i] = dists[i];
//             second_closest_center[i] = closest_center[i];
//             dists[i] = dist;
//             closest_center[i] = candidate;
//           } else if (dist < second_closest_dists[i]) {
//             second_closest_dists[i] = dist;
//             second_closest_center[i] = candidate;
//           }
//         }
//       }

//       spdlog::info("Found better centers with clustering cost {}\n", candidate_costs[min_index]);
//       clustering_cost = candidate_costs[min_index];
//     }// otherwise do nothing
//   }

//   return centers;
// }

std::vector<size_t> run(dataset_t const &data, size_t num_centers)
{// log(k) approximate k-means using k-means++, use for coreset initialization
  auto [centers, assignments, distances] = run_full(data, num_centers);

  return centers;
}

std::tuple<std::vector<size_t>, std::vector<size_t>, std::vector<coord_t>> run_full(dataset_t const &data,
  size_t num_centers)
{
  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<size_t> distrib(0, data.rows() - 1);

  auto first{ distrib(gen) };
  auto const &first_center = blaze::row(data, first);

  std::vector<size_t> centers = { first };
  std::vector<size_t> assignments(data.rows(), 0);
  std::vector<coord_t> distances(data.rows());

  // #pragma omp parallel for
  for (size_t i = 0; i < data.rows(); i++) { distances[i] = KMeans::dist(first_center, blaze::row(data, i)); }

  for (size_t i{ 1 }; i < num_centers; i++) {
    std::discrete_distribution<size_t> dist(distances.begin(), distances.end());
    auto center{ dist(gen) };
    auto const &center_point = blaze::row(data, center);
    centers.push_back(center);

    // #pragma omp parallel for
    for (size_t j = 0; j < data.rows(); j++) {
      auto distance_to_center{ KMeans::dist(center_point, blaze::row(data, j)) };
      if (distance_to_center < distances[j]) {
        assignments[j] = i;
        distances[j] = distance_to_center;
      }
    }
  }

  return std::make_tuple(centers, assignments, distances);
}
}// namespace KMeansPlusPlus
