#include "lwcoreset.hpp"

#include <algorithm>
#include <execution>
#include <random>

#include <spdlog/spdlog.h>

namespace LWCoreset {

coreset_t run(dataset_t const &data, size_t coreset_size)
{
  if (coreset_size == 0) {
    spdlog::warn("m := 0, no coreset will be computed");
    return { {}, {} };
  }

  blaze::DynamicVector<coord_t, blaze::rowVector> data_mean{ blaze::mean<blaze::columnwise>(data) };

  std::vector<coord_t> sens(data.rows());
  coord_t total{ 0.0 };

  for (size_t i{ 0 }; i < data.rows(); i++) {
    auto distance{ KMeans::dist(data_mean, blaze::row(data, i)) };
    total += distance;
    sens[i] = distance;
  }

  spdlog::info("Sensitivity vector size {}", sens.size());

  const coord_t offset{ 1.0 / static_cast<coord_t>(2 * data.rows()) };
  total *= 2.0;

  // is parallelized now
  std::transform(sens.cbegin(), sens.cend(), sens.begin(), [&offset, &total](coord_t s) { return offset + s / total; });

  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<size_t> distribution(sens.begin(), sens.end());

  std::vector<size_t> elems;
  std::vector<coord_t> weights;

  for (size_t i{ 0 }; i < coreset_size; i++) {// could probably parallelize this
    elems.push_back(distribution(gen));
    weights.push_back(1.0 / (static_cast<coord_t>(coreset_size) * sens[elems.back()]));
  }

  return { elems, weights };
}

}// namespace LWCoreset
