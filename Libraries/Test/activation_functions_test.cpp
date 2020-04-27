#include "../activation_functions.h"

#include "gtest/gtest.h"

TEST(ActivationFunctionTest,
     GivenInput_WhenLinearActivationFunction_ThenCorrctValueCalculated) {
  const double input{2.0};
  const double expected_value{2.0};

  const double actual_value = ActivationFunction::Linear(input);

  EXPECT_EQ(expected_value, actual_value);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
