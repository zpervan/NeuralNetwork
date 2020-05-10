#include "../activation_functions.h"

#include "gtest/gtest.h"

class ActivationFunctionTestFixture : public ::testing::Test {
protected:
  const std::array<const double, 4> inputs_{{1.0, 0.5, 8.45, -0.21}};
};

TEST_F(ActivationFunctionTestFixture,
       GivenInput_WhenCalculatingLinearActivationFunction_ThenResultIsCorrect) {
  const std::array<const double, 4> expected_value{1.0, 0.5, 8.45, -0.21};

  for (std::size_t i = 1; i < expected_value.size(); i++) {
    const double actual_value = ActivationFunction::Linear(inputs_.at(i));
    EXPECT_EQ(expected_value.at(i), actual_value);
  }
}

TEST_F(
    ActivationFunctionTestFixture,
    GivenInput_WhenCalculatingSigmoidActivationFunction_ThenResultIsCorrect) {
  const std::array<const double, 4> expected_value{0.73105, 0.62245, 0.99978,
                                                   0.44769};

  for (std::size_t i = 1; i < expected_value.size(); i++) {
    const double actual_value = ActivationFunction::Sigmoid(inputs_.at(i));
    EXPECT_NEAR(expected_value.at(i), actual_value, 1e-5);
  }
}

TEST_F(
    ActivationFunctionTestFixture,
    GivenInput_WhenCalculatingHyperbolicTangentActivationFunction_ThenResultIsCorrect) {
  const std::array<const double, 4> expected_values{0.10307, 0.17000, 0.00000,
                                                    -0.31499};

  for (std::size_t i = 1; i < expected_values.size(); i++) {
    const double actual_value =
        ActivationFunction::HyperbolicTangent(inputs_.at(i));
    EXPECT_NEAR(expected_values.at(i), actual_value, 1e-5);
  }
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
