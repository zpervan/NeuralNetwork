#include "../neural_network_base.h"

#include "gtest/gtest.h"

class NeuralNetworkBaseTestFixture : protected Base::NeuralNetworkBase,
                                     public ::testing::Test {
protected:
  // TODO: Add implementation
};

TEST_F(NeuralNetworkBaseTestFixture, Given_When_Then) { EXPECT_FALSE(false); }

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
