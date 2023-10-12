add_subdirectory(util)

find_package(fmt CONFIG REQUIRED)
find_package(benchmark CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(CLI11 CONFIG REQUIRED)
find_package(BLAS REQUIRED)
message("Found BLAS version ${BLAS_FOUND}")
find_package(LAPACK REQUIRED)
message("Found LAPACK version ${LAPACK_FOUND}")
find_package(blaze CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(util)

file(GLOB SOURCES "*.cpp" "util/*.cpp")

# Generic test that uses conan libs
add_executable(corset ${SOURCES})
target_link_libraries(
  corset
  PUBLIC project_options project_warnings OpenMP::OpenMP_CXX LAPACK::LAPACK
  PRIVATE CLI11::CLI11 fmt::fmt spdlog::spdlog benchmark::benchmark blaze::blaze)

target_include_directories(corset PRIVATE "${CMAKE_BINARY_DIR}/configured_files/include" src src/util)
