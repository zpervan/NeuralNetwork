#include "../neural_network_base.h"

#include <memory>

#include "gtest/gtest.h"
namespace Base {
class NeuralNetworkBaseTestFixture : protected Base::NeuralNetworkBase,
                                     public ::testing::Test {
protected:
  const std::array<Value, 4> default_input_values{{3, 2, 1, 0}};
  const std::array<Value, 2> default_values{{1.0, 2.0}};
};

TEST_F(NeuralNetworkBaseTestFixture,
       GivenTwoNeurons_WhenCreatingSynapse_ThenNeuronsAreLinked) {

  Neuron *parent_neuron{new Neuron{default_values[0]}};
  Neuron *child_neuron{new Neuron{default_values[1]}};

  auto *synapse = new Synapse(parent_neuron, child_neuron, 1.0);

  ASSERT_TRUE(synapse);

  ASSERT_NE(nullptr, synapse->GetParentNeuron());
  ASSERT_NE(nullptr, synapse->GetChildNeuron());

  const std::array<const double, 2> expected_neuron_values{1.0, 2.0};

  EXPECT_EQ(expected_neuron_values[0], synapse->GetParentNeuron()->GetValue());
  EXPECT_EQ(expected_neuron_values[1], synapse->GetChildNeuron()->GetValue());
}

TEST_F(
    NeuralNetworkBaseTestFixture,
    GivenSynapseWithParentAndChildNeuron_WhenResettingSynapse_ThenExistingNeuronsAreUnchanged) {

  const std::array<Neuron *, 2> neurons{new Neuron(default_values[0]),
                                        new Neuron(default_values[1])};

  auto *synapse = new Synapse(neurons[0], neurons[1], 0.0);

  ASSERT_TRUE(synapse);

  const std::array<double, 2> expected_value = default_values;

  EXPECT_EQ(expected_value[0], synapse->GetParentNeuron()->GetValue());
  EXPECT_EQ(expected_value[1], synapse->GetChildNeuron()->GetValue());

  const std::array<Neuron *, 2> new_neurons{new Neuron(3.0), new Neuron(4.0)};

  synapse->SetParentNeuron(new_neurons[0]);
  synapse->SetChildNeuron(new_neurons[1]);

  EXPECT_EQ(expected_value[0], synapse->GetParentNeuron()->GetValue());
  EXPECT_EQ(expected_value[1], synapse->GetChildNeuron()->GetValue());
}

TEST_F(
    NeuralNetworkBaseTestFixture,
    GivenInputValues_WhenSettingNumberOfNeuronsInInputLayer_ThenCorrectSizeIsSet) {

  const std::size_t expected_size{4};

  SetNumberOfNeuronsInInputLayer(default_input_values.size());

  ASSERT_EQ(expected_size, input_layer_.capacity());
}

TEST_F(
    NeuralNetworkBaseTestFixture,
    GivenEmptyInputValues_WhenSettingNumberOfNeuronsInInputLayer_ThenExceptionIsThrown) {

  const std::size_t invalid_size{0};

  ASSERT_THROW(SetNumberOfNeuronsInInputLayer(invalid_size),
               std::invalid_argument);
}



} // namespace Base

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
