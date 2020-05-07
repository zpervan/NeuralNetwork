#include "../feed_forward_network.h"

#include "gtest/gtest.h"

class FeedForwardNetworkTestFixture : protected FeedForwardNetwork,
                                      public ::testing::Test {};

TEST_F(
    FeedForwardNetworkTestFixture,
    GivenDefinedArchitectureAndInputValues_WhenCreatingNetwork_ThenNetworkSuccessfullyCreated) {

  const NeuralNetworkArchitecture neural_network_architecture{2, 3, 1};
  const std::vector<Value> input_values{1.0, 0.0};

  DefineNeuralNetworkArchitecture(neural_network_architecture, input_values);

  const std::size_t expected_synapses_size{9};
  ASSERT_EQ(expected_synapses_size, GetSynapses().size());
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
