#ifndef IO_H_
#define IO_H_

#include <map>
#include <optional>
#include <string>

#include <blaze/Blaze.h>

#include "utility.hpp"

namespace DataIO {
// routines to read csv files
std::optional<dataset_t> read_csv(std::string const &filename);

void export_csv(std::string const &filename,
  std::map<std::string, std::map<size_t, std::vector<double>>> const &benchmark_results);

void export_vector(std::string const &filename, std::vector<coord_t> const &vec);

void export_dataset_to_csv(std::string const &filename, dataset_t const &data);

void export_cluster_costs(std::string const &filename, std::map<std::string, coord_t> const &clustering_costs);

void export_coreset(dataset_t const &data, coreset_t const &coreset, std::string const &filename);
};// namespace DataIO

#endif// IO_H_
