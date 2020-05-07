#include "../exclusive_or_neural_network.h"
#include "gtest/gtest.h"

class ExclusiveOrNeuralNetworkTestFixture : protected ExclusiveOrNeuralNetwork,
                                            public ::testing::Test {
protected:
  const NeuralNetworkArchitecture neural_network_architecture_{2, 3, 1};
  const std::vector<Value> input_values_{1.0, 0.0};
  const std::size_t expected_synapses_size_{9};

  void CheckSynapsesChildNeuronValuesNotZero(
      const std::multimap<Id, Synapse> &actual_values) {
    for (const auto &actual_value : actual_values) {
      EXPECT_NE(0, actual_value.second.GetChildNeuron()->GetValue())
          << "Synapse ID: " << actual_value.first << "\n";
    }
  };
};

TEST_F(
    ExclusiveOrNeuralNetworkTestFixture,
    GivenDefinedArchitectureAndInputValues_WhenCreatingNetwork_ThenNetworkSuccessfullyCreated) {

  DefineNeuralNetworkArchitecture(neural_network_architecture_, input_values_);
  ASSERT_EQ(expected_synapses_size_, GetSynapses().size());

  CalculateInitialValues();

  CheckSynapsesChildNeuronValuesNotZero(synapses_);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
