#include "io.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>

#include <spdlog/spdlog.h>

namespace DataIO {

class csv_row
{
public:
  csv_row() : m_separator(','), m_line(""), m_positions() {}
  explicit csv_row(char separator) : m_separator(separator), m_line(""), m_positions(){};

  std::string_view operator[](size_t index)
  {
    return { &m_line[m_positions[index] + 1], m_positions[index + 1] - (m_positions[index] + 1) };
  }

  [[nodiscard]] size_t size() const { return m_positions.size() - 1; }

  void read_next_line(std::istream &str)
  {
    // read a new line, skipping whitespace lines, usually these are trailing
    do {
      std::getline(str, m_line);
    } while (std::all_of(m_line.begin(), m_line.end(), [](unsigned char const current) {
      return std::isspace(current);
    }) && !m_line.empty());
    m_positions.clear();
    m_positions.emplace_back(-1);
    std::string::size_type pos = 0;

    while ((pos = m_line.find(m_separator, pos)) != std::string::npos) {
      m_positions.emplace_back(pos);
      ++pos;
    }

    pos = m_line.size();
    m_positions.emplace_back(pos);
  }

private:
  char m_separator;
  std::string m_line;
  std::vector<size_t> m_positions;
};

std::istream &operator>>(std::istream &str, csv_row &data)
{
  data.read_next_line(str);
  return str;
}


std::optional<dataset_t> read_csv(std::string const &filename)
{
  std::ifstream file(filename);
  csv_row row;

  size_t rows{ 0 };

  std::vector<coord_t> points;


  if (file) {
    while (file >> row) {
      for (size_t i{ 0 }; i < row.size(); i++) {
        auto stringview{ row[i] };
        char *end;
        coord_t const number{ std::strtod(stringview.data(), &end) };

        if (end == stringview.data()) {
          spdlog::warn("Could not parse string to number: {}", stringview.data());
          spdlog::warn("Parsed number is: {}", number);
        } else {

          points.emplace_back(number);
        }
      }
      rows++;
    }

    size_t cols{ points.size() / rows };

    blaze::DynamicMatrix<coord_t, blaze::rowMajor> dataset(rows, cols, points.data());

    return dataset;

  } else {
    spdlog::error("File could not be opened!");
    spdlog::error("Error: {}", std::strerror(errno));
    return {};
  }
}

// used to export a map of maps, invoked when exporting clustering results, where the dataset is structured as Name ->
// (k -> Costs)
void export_csv(std::string const &filename,
  std::map<std::string, std::map<size_t, std::vector<double>>> const &benchmark_results)
{
  std::ofstream file(filename);

  for (auto [alg_name, res] : benchmark_results) {
    for (auto [num_centers, clustering_costs] : res) {
      for (auto clustering_cost : clustering_costs) {
        file << alg_name << "," << std::to_string(num_centers) << "," << std::to_string(clustering_cost) << std::endl;
      }
    }
  }
}

// write out a vector of floating point numbers
void export_vector(std::string const &filename, std::vector<coord_t> const &vec)
{
  std::ofstream file(filename);
  for (auto const &elem : vec) { file << std::to_string(elem) << std::endl; }
}

void export_dataset_to_csv(std::string const &filename, dataset_t const &data)
{
  std::ofstream file(filename);
  for (size_t i{ 0 }; i < data.rows(); i++) {
    auto point{ blaze::row(data, i) };

    for (auto it{ point.begin() }; it < point.end(); it++) {
      file << *it;
      if (it + 1 != point.end()) { file << ','; }
    }
    file << std::endl;
  }
}

void export_cluster_costs(std::string const &filename, std::map<std::string, coord_t> const &clustering_costs)
{
  std::ofstream file(filename);

  for (auto [name, costs] : clustering_costs) { file << name << "," << std::to_string(costs) << std::endl; }
}

void export_coreset(dataset_t const &data, coreset_t const &coreset, std::string const &filename)
{
  std::ofstream file(filename);
  auto const &[points, weights] = coreset;
  auto coreset_size = points.size();
  spdlog::info("Writing coreset to {}", filename);

  for (size_t i{ 0 }; i < coreset_size; i++) {
    auto const &point = blaze::row(data, points[i]);
    for (auto coordinate : point) {
      // print the coordinate
      file << std::to_string(coordinate) << ",";
    }
    file << weights[i] << std::endl;
  }
  spdlog::info("Finished writing {}. Wrote {} bytes", filename, file.tellp());
}
};// namespace DataIO
