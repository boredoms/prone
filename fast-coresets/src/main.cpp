#include <algorithm>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <functional>
#include <future>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

#include <CLI/CLI.hpp>
#include <benchmark/benchmark.h>
#include <blaze/Math.h>
#include <omp.h>
#include <spdlog/spdlog.h>

#include "coreset.hpp"
#include "generators.hpp"
#include "io.hpp"
#include "kmeans1d.hpp"
#include "kmeanspp.hpp"
#include "lwcoreset.hpp"
#include "utility.hpp"

#include <internal_use_only/config.hpp>

constexpr size_t DEFAULT_NUM_RUNS{ 5 };// default number of runs

// compute the clustering costs on a dataset using the closest center assignment
void clustering_benchmark(std::string filename, size_t num_runs)
{
  spdlog::info("Running clustering benchmark on file {}", filename);
  spdlog::info("Reading dataset...");

  auto dataset = DataIO::read_csv(filename);
  std::string dataset_name{ std::filesystem::path(filename).stem() };

  std::vector<size_t> centers{ 10, 25, 50, 100, 250, 500, 1000, 2500, 5000 };

  std::map<size_t, std::vector<double>> cost_fast_kmeans;
  std::map<size_t, std::vector<double>> cost_fast_kmeans_biased;
  std::map<size_t, std::vector<double>> cost_fast_kmeans_covariance;
  std::map<size_t, std::vector<double>> cost_fast_kmeans_assignment;
  std::map<size_t, std::vector<double>> cost_kmeans;
  std::map<size_t, std::vector<double>> cost_random;

  std::map<std::string, double> cluster_timings;
  std::map<std::string, double> cluster_timings_with_assignment;

  for (auto num_centers : centers) {
    spdlog::info("Running k-means clustering algorithms with k = {}", num_centers);

    for (size_t i{ 0 }; i < num_runs; i++) {
      spdlog::info("k-means++");

      auto t1 = std::chrono::high_resolution_clock::now();
      auto [centers, assignments, distances] = KMeansPlusPlus::run_full(*dataset, num_centers);
      auto t2 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> ms = t2 - t1;

      spdlog::info("Computing the clustering took {} ms", ms.count());
      cluster_timings[fmt::format("kmeans++-{}-{}-{}", dataset_name, num_centers, i + 1)] = ms.count();

      auto const kmeans_cost = std::reduce(distances.cbegin(), distances.cend());
      t2 = std::chrono::high_resolution_clock::now();
      ms = t2 - t1;
      cluster_timings_with_assignment[fmt::format("kmeans++-{}-{}-{}", dataset_name, num_centers, i + 1)] = ms.count();

      fmt::print("Clustering cost (k-means++): {}\n", kmeans_cost);
      cost_kmeans[num_centers].emplace_back(kmeans_cost);

      {
        spdlog::info("fast-k-means++");

        t1 = std::chrono::high_resolution_clock::now();
        auto [found_fast_centers, fast_assignments, fast_distances] =
          FastKMeans::efficient_k_means_full(*dataset, num_centers);
        t2 = std::chrono::high_resolution_clock::now();
        ms = t2 - t1;

        spdlog::info("Computing the clustering took {} ms", ms.count());
        cluster_timings[fmt::format("fastkmeans++-{}-{}-{}", dataset_name, num_centers, i + 1)] = ms.count();

        auto const fast_cost = KMeans::clustering_cost(*dataset, found_fast_centers);
        auto const fast_cost_assignment = std::reduce(fast_distances.cbegin(), fast_distances.cend());
        t2 = std::chrono::high_resolution_clock::now();
        ms = t2 - t1;

        spdlog::info("Computing the clustering took {} ms", ms.count());
        cluster_timings_with_assignment[fmt::format("fastkmeans++-{}-{}-{}", dataset_name, num_centers, i + 1)] =
          ms.count();
        fmt::print("Clustering cost (k-means++ fast): {}\n", fast_cost);
        cost_fast_kmeans[num_centers].emplace_back(fast_cost);
      }

      {
        spdlog::info("fast-k-means++-biased");

        t1 = std::chrono::high_resolution_clock::now();
        auto [found_fast_centers, fast_assignments, fast_distances] =
          FastKMeans::efficient_k_means_full_biased(*dataset, num_centers);
        t2 = std::chrono::high_resolution_clock::now();
        ms = t2 - t1;

        spdlog::info("Computing the clustering took {} ms", ms.count());
        cluster_timings[fmt::format("fastkmeans++biased-{}-{}-{}", dataset_name, num_centers, i + 1)] = ms.count();

        auto const fast_cost = KMeans::clustering_cost(*dataset, found_fast_centers);
        auto const fast_cost_assignment = std::reduce(fast_distances.cbegin(), fast_distances.cend());
        t2 = std::chrono::high_resolution_clock::now();
        ms = t2 - t1;

        spdlog::info("Computing the clustering took {} ms", ms.count());
        cluster_timings_with_assignment[fmt::format("fastkmeans++-{}-{}-{}", dataset_name, num_centers, i + 1)] =
          ms.count();
        fmt::print("Clustering cost (k-means++ fast): {}\n", fast_cost);
        cost_fast_kmeans_biased[num_centers].emplace_back(fast_cost);
      }

      {
        spdlog::info("fast-k-means++-covariance");

        t1 = std::chrono::high_resolution_clock::now();
        auto [found_fast_centers, fast_assignments, fast_distances] =
          FastKMeans::efficient_k_means_full_covariance(*dataset, num_centers);
        t2 = std::chrono::high_resolution_clock::now();
        ms = t2 - t1;

        spdlog::info("Computing the clustering took {} ms", ms.count());
        cluster_timings[fmt::format("fastkmeans++covariance-{}-{}-{}", dataset_name, num_centers, i + 1)] = ms.count();

        auto const fast_cost = KMeans::clustering_cost(*dataset, found_fast_centers);
        auto const fast_cost_assignment = std::reduce(fast_distances.cbegin(), fast_distances.cend());
        t2 = std::chrono::high_resolution_clock::now();
        ms = t2 - t1;

        spdlog::info("Computing the clustering took {} ms", ms.count());
        cluster_timings_with_assignment[fmt::format("fastkmeans++-{}-{}-{}", dataset_name, num_centers, i + 1)] =
          ms.count();
        fmt::print("Clustering cost (k-means++ fast): {}\n", fast_cost);
        cost_fast_kmeans_covariance[num_centers].emplace_back(fast_cost);
      }
    }
  }
  std::map<std::string, std::map<size_t, std::vector<double>>> results;
  results["k-means++"] = cost_kmeans;
  results["Ours"] = cost_fast_kmeans;
  results["Ours (biased)"] = cost_fast_kmeans_biased;
  results["Ours (covariance)"] = cost_fast_kmeans_covariance;

  DataIO::export_cluster_costs(fmt::format("cluster-timings-{}.csv", dataset_name), cluster_timings);
  DataIO::export_cluster_costs(
    fmt::format("cluster-timings-assignment-{}.csv", dataset_name), cluster_timings_with_assignment);

  std::async(std::launch::async, DataIO::export_csv, fmt::format("clustering_results_{}.csv", dataset_name), results);
}

// clustering benchmark using the assignments produced by the algorithms
void clustering_benchmark_assignment(std::string filename, size_t num_runs)
{
  spdlog::info("Running clustering benchmark on file {}", filename);
  spdlog::info("Reading dataset...");

  auto dataset = DataIO::read_csv(filename);
  std::string dataset_name{ std::filesystem::path(filename).stem() };

  std::vector<size_t> centers{ 10, 25, 50, 100, 250, 500, 1000, 2500, 5000 };

  std::map<size_t, std::vector<double>> cost_fast_kmeans;
  std::map<size_t, std::vector<double>> cost_fast_kmeans_biased;
  std::map<size_t, std::vector<double>> cost_fast_kmeans_covariance;

  std::map<std::string, double> cluster_timings;

  for (auto num_centers : centers) {
    spdlog::info("Running k-means clustering algorithms with k = {}", num_centers);

    for (size_t i{ 0 }; i < num_runs; i++) {
      spdlog::info("k-means++");

      auto t1 = std::chrono::high_resolution_clock::now();
      auto t2 = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double, std::milli> ms = t2 - t1;

      {
        spdlog::info("fast-k-means++");

        t1 = std::chrono::high_resolution_clock::now();
        auto [found_fast_centers, fast_assignments, fast_distances] =
          FastKMeans::efficient_k_means_full(*dataset, num_centers);
        t2 = std::chrono::high_resolution_clock::now();
        ms = t2 - t1;

        spdlog::info("Computing the clustering took {} ms", ms.count());
        cluster_timings[fmt::format("fastkmeans++-{}-{}-{}", dataset_name, num_centers, i + 1)] = ms.count();

        auto const fast_cost = std::reduce(fast_distances.begin(), fast_distances.end());
        fmt::print("Clustering cost (k-means++ fast): {}\n", fast_cost);
        cost_fast_kmeans[num_centers].emplace_back(fast_cost);
      }

      {
        spdlog::info("fast-k-means++-biased");

        t1 = std::chrono::high_resolution_clock::now();
        auto [found_fast_centers, fast_assignments, fast_distances] =
          FastKMeans::efficient_k_means_full_biased(*dataset, num_centers);
        t2 = std::chrono::high_resolution_clock::now();
        ms = t2 - t1;

        spdlog::info("Computing the clustering took {} ms", ms.count());
        cluster_timings[fmt::format("fastkmeans++biased-{}-{}-{}", dataset_name, num_centers, i + 1)] = ms.count();

        auto const fast_cost = std::reduce(fast_distances.begin(), fast_distances.end());
        fmt::print("Clustering cost (k-means++ fast): {}\n", fast_cost);
        cost_fast_kmeans_biased[num_centers].emplace_back(fast_cost);
      }

      {
        spdlog::info("fast-k-means++-covariance");

        t1 = std::chrono::high_resolution_clock::now();
        auto [found_fast_centers, fast_assignments, fast_distances] =
          FastKMeans::efficient_k_means_full_covariance(*dataset, num_centers);
        t2 = std::chrono::high_resolution_clock::now();
        ms = t2 - t1;

        spdlog::info("Computing the clustering took {} ms", ms.count());
        cluster_timings[fmt::format("fastkmeans++covariance-{}-{}-{}", dataset_name, num_centers, i + 1)] = ms.count();

        auto const fast_cost = std::reduce(fast_distances.begin(), fast_distances.end());
        fmt::print("Clustering cost (k-means++ fast): {}\n", fast_cost);
        cost_fast_kmeans_covariance[num_centers].emplace_back(fast_cost);
      }
    }
  }
  std::map<std::string, std::map<size_t, std::vector<double>>> results;
  results["fast-k-means++"] = cost_fast_kmeans;
  results["fast-k-means++-biased"] = cost_fast_kmeans_biased;
  results["fast-k-means++-covariance"] = cost_fast_kmeans_covariance;

  DataIO::export_cluster_costs(fmt::format("cluster-timings-{}.csv", dataset_name), cluster_timings);

  std::async(std::launch::async, DataIO::export_csv, fmt::format("clustering_results_{}.csv", dataset_name), results);
}

void coreset_benchmark(std::string filename, size_t const num_runs)
{
  spdlog::info("Running coreset benchmark on file {}", filename);

  spdlog::info("Reading dataset...", filename);
  auto dataset = DataIO::read_csv(filename);
  std::string dataset_name{ std::filesystem::path(filename).stem() };

  std::vector<size_t> centers{ 10, 100, 1000 };
  std::vector<coord_t> relative_coreset_sizes{ 0.0001, 0.00025, 0.0005 };
  //, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1 };

  std::map<std::string, double> coreset_timings;


  for (size_t i = 0; i < num_runs; i++) {
    for (size_t j = 0; j < relative_coreset_sizes.size(); j++) {
      for (auto num_centers : centers) {
        auto multiplier{ relative_coreset_sizes[j] };
        size_t size{ static_cast<size_t>(multiplier * (*dataset).rows()) };
        if (size < num_centers) { continue; }
        spdlog::info("Computing coresets of size {}", size);
        {
          spdlog::info("Fast Coreset (ours)");
          auto t1 = std::chrono::high_resolution_clock::now();
          auto fast_coreset = SensitivitySampling::run_near_linear(*dataset, num_centers, size);
          auto t2 = std::chrono::high_resolution_clock::now();
          std::chrono::duration<double, std::milli> ms = t2 - t1;
          spdlog::info("Computing the coreset took {} ms", ms.count());
          coreset_timings[fmt::format("Ours-{}-{}-{}-{}", dataset_name, num_centers, size, i + 1)] = ms.count();
          spdlog::info("Exporting...");
          DataIO::export_coreset(*dataset,
            fast_coreset,
            fmt::format("coresets/Ours-{}-{}-{}-{}.csv", dataset_name, num_centers, size, i + 1));
        }
        // {
        //   spdlog::info("Fast Coreset (biased)");
        //   auto t1 = std::chrono::high_resolution_clock::now();
        //   auto fast_coreset = SensitivitySampling::run_near_linear_biased(*dataset, num_centers, size);
        //   auto t2 = std::chrono::high_resolution_clock::now();
        //   std::chrono::duration<double, std::milli> ms = t2 - t1;
        //   spdlog::info("Computing the coreset took {} ms", ms.count());
        //   coreset_timings[fmt::format("biased-{}-{}-{}-{}", dataset_name, num_centers, size, i + 1)] = ms.count();
        //   spdlog::info("Exporting...");
        //   DataIO::export_coreset(*dataset,
        //     fast_coreset,
        //     fmt::format("coresets/biased-{}-{}-{}-{}.csv", dataset_name, num_centers, size, i + 1));
        // }
        // {
        //   spdlog::info("Fast Coreset (covariance)");
        //   auto t1 = std::chrono::high_resolution_clock::now();
        //   auto fast_coreset = SensitivitySampling::run_near_linear_covariance(*dataset, num_centers, size);
        //   auto t2 = std::chrono::high_resolution_clock::now();
        //   std::chrono::duration<double, std::milli> ms = t2 - t1;
        //   spdlog::info("Computing the coreset took {} ms", ms.count());
        //   coreset_timings[fmt::format("covariance-{}-{}-{}-{}", dataset_name, num_centers, size, i + 1)] =
        //   ms.count(); spdlog::info("Exporting..."); DataIO::export_coreset(*dataset,
        //     fast_coreset,
        //     fmt::format("coresets/covariance-{}-{}-{}-{}.csv", dataset_name, num_centers, size, i + 1));
        // }

        spdlog::info("Lightweight Coreset");
        auto t1 = std::chrono::high_resolution_clock::now();
        auto lw_coreset = LWCoreset::run(*dataset, size);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms = t2 - t1;
        spdlog::info("Computing the coreset took {} ms", ms.count());
        coreset_timings[fmt::format("Lightweight-{}-{}-{}-{}", dataset_name, num_centers, size, i + 1)] = ms.count();

        spdlog::info("Lightweight Coreset");
        DataIO::export_coreset(*dataset,
          lw_coreset,
          fmt::format("coresets/Lightweight-{}-{}-{}-{}.csv", dataset_name, num_centers, size, i + 1));
      }
    }
  }

  for (size_t i = 0; i < num_runs; i++) {
    for (size_t j = 0; j < relative_coreset_sizes.size(); j++) {
      for (auto num_centers : centers) {
        auto multiplier{ relative_coreset_sizes[j] };
        size_t size{ static_cast<size_t>(multiplier * (*dataset).rows()) };
        if (size < num_centers) { continue; }
        spdlog::info("Computing coresets of size {}", size);

        spdlog::info("Sensitivity Sampling Coreset");
        auto t1 = std::chrono::high_resolution_clock::now();
        auto kmeans_coreset = SensitivitySampling::run(*dataset, num_centers, size);
        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms = t2 - t1;
        spdlog::info("Computing the coreset took {} ms", ms.count());
        coreset_timings[fmt::format("Sensitivity-{}-{}-{}-{}", dataset_name, num_centers, size, i + 1)] = ms.count();

        spdlog::info("Exporting coresets");
        spdlog::info("Sensitivity Sampling Coreset");
        DataIO::export_coreset(*dataset,
          kmeans_coreset,
          fmt::format("coresets/Sensitivity-{}-{}-{}-{}.csv", dataset_name, num_centers, size, i + 1));
      }
    }
  }

  DataIO::export_cluster_costs(fmt::format("coreset-timings-{}.csv", dataset_name), coreset_timings);
}

// NOLINTNEXTLINE(bugprone-exception-escape)
int main(int argc, const char **argv)
{
  try {
    // set up the CLI argument passing
    CLI::App app{ fmt::format("{} version {}", myproject::cmake::project_name, myproject::cmake::project_version) };

    std::optional<std::string> filename;
    std::optional<std::string> task;
    std::optional<size_t> centers;

    // [Cluster]
    auto cli_cluster{ app.add_subcommand("cluster", "Cluster a dataset") };
    cli_cluster->add_option("file", filename, "Filename of the dataset");
    cli_cluster->add_option("centers", centers, "Number of centers");
    cli_cluster->callback([&]() {
      spdlog::info("Computing a coreset of the file {}", *filename);
      auto dataset = DataIO::read_csv(*filename);

      spdlog::info("Running fast-k-means++");
      auto [found_centers, assignments, distances] = FastKMeans::efficient_k_means_full(*dataset, *centers);

      spdlog::info("The found centers are:");
      for (auto center : found_centers) { spdlog::info(" {}", center); }

      spdlog::info("The points are assigned to:");
      for (auto center : assignments) { spdlog::info(" {}", center); }

      spdlog::info("The distances are:");
      for (auto distance : distances) { spdlog::info(" {}", distance); }
    });

    // [Coreset]
    auto cli_coreset{ app.add_subcommand("coreset", "Compute a coreset from a dataset") };
    std::optional<size_t> coreset_size;

    cli_coreset->add_option("file", filename, "Filename of the dataset");
    cli_coreset->add_option("centers", centers, "Number of centers");
    cli_coreset->add_option("size", coreset_size, "Size of coreset");
    cli_coreset->callback([&]() {
      spdlog::info("Computing a coreset of the file {}", *filename);
      auto dataset = DataIO::read_csv(*filename);
      auto fast_coreset = SensitivitySampling::run_near_linear(*dataset, *centers, *coreset_size);
      DataIO::export_coreset(*dataset, fast_coreset, "coreset.csv");
    });

    // [Score]
    auto cli_score{ app.add_subcommand("score", "Compute the clustering cost of a set of centroids in a dataset") };
    cli_score->add_option("file", filename, "Filename of the dataset");
    std::vector<std::string> filename_centroids;
    cli_score->add_option("centroids", filename_centroids, "Filename of the file storing the computed centroids");
    cli_score->callback([&]() {
      spdlog::info("Scoring the centroids against the dataset {}", *filename);

      spdlog::info("Reading the dataset.");
      auto dataset = DataIO::read_csv(*filename);

      std::map<std::string, coord_t> cost_map;
      size_t counter{ 1 };
      size_t total{ filename_centroids.size() };

#pragma omp parallel for schedule(dynamic)
      for (auto &cluster_file : filename_centroids) {
#pragma omp critical
        spdlog::info("Scoring {}, ({} of {})", cluster_file, counter++, total);

        auto centroids = DataIO::read_csv(cluster_file);
        auto centroid_clustering_cost{ KMeans::clustering_cost(*dataset, *centroids) };

#pragma omp critical
        cost_map[cluster_file] = centroid_clustering_cost;
      }

      std::string filename_stem{ std::filesystem::path(*filename).stem() };
      DataIO::export_cluster_costs(fmt::format("cluster-costs-{}.csv", filename_stem), cost_map);
    });

    // Statistics
    auto cli_stats{ app.add_subcommand("stats", "Compute statistics") };
    auto cli_dists{ cli_stats->add_subcommand("dists", "Compute distances to the closest center") };
    cli_dists->add_option("file", filename, "Filename of the dataset");
    std::vector<std::string> dists_centroids;
    cli_dists->add_option("centroids", dists_centroids, "Filename of the centroids");
    cli_dists->callback([&]() {
      auto data = DataIO::read_csv(*filename);
      for (auto centroids_file : dists_centroids) {
        std::string centroids_stem{ std::filesystem::path(centroids_file).stem() };
        auto centroids = DataIO::read_csv(centroids_file);
        auto dists = KMeans::dists(*data, *centroids);

        DataIO::export_vector(fmt::format("point-costs-{}.txt", centroids_stem), dists);
      }
    });


    // [Benchmark]
    auto cli_benchmark{ app.add_subcommand("benchmark", "Run the benchmarks on a dataset") };
    std::optional<size_t> num_runs;// number of times to run the benchmark
    std::vector<std::string> filenames;

    // [Benchmark->Clustering]
    auto cli_benchmark_clustering{ cli_benchmark->add_subcommand("clustering", "Compare fast clustering algorithms") };
    cli_benchmark_clustering->add_option("runs", num_runs, "Number of runs");
    cli_benchmark_clustering->add_option("file", filenames, "Filenames of the datasets to benchmark on");

    cli_benchmark_clustering->callback([&]() {
      if (filenames.empty()) { spdlog::warn("No files provided. Exiting."); }
      if (!num_runs) { num_runs = DEFAULT_NUM_RUNS; }
      for (auto const &dataset_filename : filenames) { clustering_benchmark(dataset_filename, *num_runs); }
    });

    // [Benchmark->FastClustering]
    auto cli_benchmark_assignment{ cli_benchmark->add_subcommand(
      "fast-clustering", "Compare fast clustering algorithms") };
    cli_benchmark_assignment->add_option("runs", num_runs, "Number of runs");
    cli_benchmark_assignment->add_option("file", filenames, "Filenames of the datasets to benchmark on");

    cli_benchmark_assignment->callback([&]() {
      if (filenames.empty()) { spdlog::warn("No files provided. Exiting."); }
      if (!num_runs) { num_runs = DEFAULT_NUM_RUNS; }
      for (auto const &dataset_filename : filenames) { clustering_benchmark_assignment(dataset_filename, *num_runs); }
    });

    // [Benchmark->Coreset]
    auto cli_benchmark_coreset{ cli_benchmark->add_subcommand("coreset", "Compare coreset algorithms") };
    // cli_benchmark_coreset->add_option("centers", centers, "Number of centers");
    cli_benchmark_coreset->add_option("runs", num_runs, "Number of runs");
    cli_benchmark_coreset->add_option("file", filenames, "Filename of the datasets to generate coresets for");

    cli_benchmark_coreset->callback([&]() {
      if (filenames.empty()) { spdlog::warn("No files provided. Exiting."); }
      if (!num_runs) { num_runs = DEFAULT_NUM_RUNS; }
      for (auto const &dataset_filename : filenames) { coreset_benchmark(dataset_filename, *num_runs); }
    });

    // [Benchmark->Coreset-Time]
    auto cli_benchmark_coreset_time{ cli_benchmark->add_subcommand(
      "coreset-time", "Compare runtime of coreset constructions") };
    cli_benchmark_coreset_time->add_option("runs", num_runs, "Number of runs");

    // [Generate]
    auto cli_generate{ app.add_subcommand("generate", "Generate synthetic datasets.") };

    // [Generate->Symmetric]
    auto cli_generate_symmetric{ cli_generate->add_subcommand(
      "symmetric", "Generate a dataset that LW coresets do badly on.") };

    std::optional<size_t> symmetric_num_inner;
    std::optional<size_t> symmetric_num_outer;
    std::optional<size_t> symmetric_num_dimensions;
    std::optional<coord_t> symmetric_scale;

    cli_generate_symmetric->add_option("num_inner", symmetric_num_inner, "Number of points in inner cluster.");
    cli_generate_symmetric->add_option("num_outer", symmetric_num_outer, "Number of points in outer clusters.");
    cli_generate_symmetric->add_option(
      "num_dimensions", symmetric_num_dimensions, "Number of dimensions of the dataset.");
    cli_generate_symmetric->add_option("scale", symmetric_scale, "Distance of clusters from origin.");

    cli_generate_symmetric->callback([&]() {
      spdlog::info("Generating dataset of size {} by {}",
        *symmetric_num_inner + 2 * *symmetric_num_dimensions * *symmetric_num_outer,
        *symmetric_num_dimensions);

      auto data = Generator::symmetric_dataset(
        *symmetric_num_inner, *symmetric_num_outer, *symmetric_num_dimensions, *symmetric_scale);
      spdlog::info("Exporting dataset");
      DataIO::export_dataset_to_csv("data/symmetric.csv", data);
    });

    // [Version]
    bool show_version = false;
    app.add_flag("--version", show_version, "Show version information");

    // Use the default logger (stdout, multi-threaded, colored)
    spdlog::info("Hewwo from the k-Means clustering utility owo");
    spdlog::info("Number of OMP threads: {}", omp_get_max_threads());

    // now parse the arguments and run the actual utility
    CLI11_PARSE(app, argc, argv);

    if (show_version) {
      fmt::print("{} version {}\n", myproject::cmake::project_name, myproject::cmake::project_version);
      return EXIT_SUCCESS;
    }
  } catch (const std::exception &e) {
    spdlog::error("Unhandled exception in main: {}", e.what());
  }

  return EXIT_SUCCESS;
}
