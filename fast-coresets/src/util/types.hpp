#ifndef TYPES_H_
#define TYPES_H_
#pragma once

#include <vector>

#include <blaze/Blaze.h>
#include <spdlog/spdlog.h>

// This file defines the basic types used in the project

using coord_t = double;
using size_t = std::size_t;
using matrix_t = blaze::DynamicMatrix<coord_t, blaze::rowMajor>;
using dataset_t = matrix_t;
using dataset_1D_t = blaze::DynamicVector<coord_t>;// point_t;
using centers_t = std::vector<size_t>;// list of indices of centers

struct coreset_t
{
  centers_t points;
  std::vector<coord_t> weights;

  coreset_t(centers_t ps, std::vector<coord_t> ws) : points{ ps }, weights{ ws }
  {
    if (points.size() != weights.size()) {
      spdlog::error("Mismatched size between points and weights: {} != {}", points.size(), weights.size());
      throw std::invalid_argument("Mismatched lengths of arguments.");
    }
  }
};

#endif// TYPES_H_
