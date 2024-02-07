#include <gtest/gtest.h>

#include <iostream>
#include <numeric>
#include <vector>

#include "pronelib.cpp"

TEST(SamplerTests, SizeTest1) {
  std::vector<double> data{0};
  sampler sampler(data.data(), data.size(), 0);

  EXPECT_EQ(sampler.size(), 1);
  EXPECT_EQ(sampler.num_points(), 1);
}

TEST(SamplerTests, SizeTest2) {
  std::vector<double> data{0, 1, 2, 3};
  sampler sampler(data.data(), data.size(), 0);

  EXPECT_EQ(sampler.size(), 7);
  EXPECT_EQ(sampler.num_points(), data.size());
}

TEST(SamplerTests, SizeTest3) {
  std::vector<double> data{0, 1, 2, 3, 4};
  sampler sampler(data.data(), data.size(), 0);

  EXPECT_EQ(sampler.size(), 15);
  EXPECT_EQ(sampler.num_points(), data.size());
}

TEST(SamplerTests, SumTest1) {
  std::vector<double> data{10};

  sampler sampler(data.data(), data.size(), 0);

  EXPECT_DOUBLE_EQ(sampler.sum(), 0);
}

TEST(SamplerTests, SumTest2) {
  std::vector<double> data{0, 1, 2, 3, 4};
  auto sum = std::inner_product(data.begin(), data.end(), data.begin(), 0);

  sampler sampler(data.data(), data.size(), 0);

  EXPECT_DOUBLE_EQ(sampler.sum(), sum);
}

TEST(SamplerTests, SumTest3) {
  std::vector<double> data{0, 1, 2, 3};

  sampler sampler(data.data(), data.size(), 1);

  EXPECT_DOUBLE_EQ(sampler.sum(), 6);
}

TEST(SamplerTests, FindTest1) {
  std::vector<double> data{0, 0, 1, 0, 0};

  sampler sampler(data.data(), data.size(), 0);

  EXPECT_EQ(sampler.find(0.5), 2);
}

TEST(SamplerTests, FindTest2) {
  std::vector<double> data{0, 1, 0, 0, 0, 1, 0, 0, 0, 1};

  sampler sampler(data.data(), data.size(), 0);

  EXPECT_EQ(sampler.find(0.5), 1);
  EXPECT_EQ(sampler.find(1.5), 5);
  EXPECT_EQ(sampler.find(2.5), 9);
}

TEST(SamplerTests, FindTestExcept) {
  std::vector<double> data{0, 1, 100, 0, 13, 0, 1};

  sampler sampler(data.data(), data.size(), 0);

  EXPECT_ANY_THROW(sampler.find(sampler.sum() + 1));
}

TEST(SamplerTests, UpdateTest1) {
  std::vector<double> data{0, 1, 0, 0, 0, 1, 0, 0, 0, 1};

  sampler sampler(data.data(), data.size(), 0);
  sampler.update(1, data.data());

  EXPECT_DOUBLE_EQ(sampler.sum(), 2);
  EXPECT_EQ(sampler.find(0.5), 5);
}

TEST(SamplerTests, UpdateTest2) {
  std::vector<double> data{0, 1, 1, 1, 1, 1, 0, 0, 1};

  sampler sampler(data.data(), data.size(), 0);
  sampler.update(1, data.data());

  EXPECT_DOUBLE_EQ(sampler.sum(), 1);
  EXPECT_EQ(sampler.find(0.5), 8);
}

TEST(ProneTests, ProneTest1) {
  int k = 2;

  std::vector<double> data{1, 1, 0, 0, 1, 1, 1, 0, 1};
  std::vector<int> centers(k, 0), assignments(data.size(), 0);

  ProneKernel pk;

  pk.run(data.data(), data.size(), k, centers.data(), assignments.data());

  auto zero_count = std::count(assignments.begin(), assignments.end(), 0);
  auto one_count = std::count(assignments.begin(), assignments.end(), 1);

  std::cout << "centers: " << centers[0] << ", " << centers[1];
  std::cout << "zero: " << zero_count << " one: " << one_count << std::endl;

  EXPECT_TRUE(zero_count == 3 && one_count == 6 ||
              zero_count == 6 && one_count == 3);
}

TEST(ProneTests, ProneTest2) {
  int k = 4;

  std::vector<double> data{1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<int> centers(k, 0), assignments(data.size(), 0);

  ProneKernel pk;

  pk.run(data.data(), data.size(), k, centers.data(), assignments.data());

  bool all_present = true;

  std::cout << "Centers: ";
  for (auto c : centers) {
    std::cout << c << " ";
  }
  std::cout << std::endl;

  for (auto c : {0, 1, 2}) {
    auto present = false;
    for (auto i = 0; i < k; i++) {
      if (centers[i] == c) {
        present = true;
      }
    }
    all_present = all_present && present;
  }

  EXPECT_TRUE(all_present);
}
