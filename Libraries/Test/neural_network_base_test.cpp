#include "../neural_network_base.h"

#include <memory>

#include "gtest/gtest.h"

class NeuralNetworkBaseTestFixture : protected NeuralNetworkBase,
                                     public ::testing::Test {
protected:
  const std::vector<Value> default_input_values{3.0, 2.0, 1.0, 0.0};
  const std::array<Value, 2> default_values{{1.0, 2.0}};

  void CreateLayersWithDefaultSize() {
    // Input layer size will be defined in input value assignment
    SetNumberOfNeuronsInInputLayer(default_input_values.size());

    ASSERT_EQ(default_input_values.size(), input_layer_.capacity());

    const std::size_t hidden_layer_size{5};
    SetNumberOfNeuronsInSingleHiddenLayer(hidden_layer_size);

    ASSERT_EQ(hidden_layer_size, hidden_layer_.capacity());

    const std::size_t output_layer_size{2};
    SetNumberOfNeuronsInOutputLayer(output_layer_size);

    ASSERT_EQ(output_layer_size, output_layer_.capacity());
  }
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
    GivenSynapseWithParentAndChildNeuron_WhenResettingSynapse_ThenCurrentSynapseUnchanged) {

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

TEST_F(NeuralNetworkBaseTestFixture,
       GivenValidInputValues_WhenSettingLayersSize_ThenCorrectSizeIsSet) {

  const std::size_t expected_size{4};
  const std::size_t layer_size = default_input_values.size();

  SetNumberOfNeuronsInInputLayer(layer_size);
  ASSERT_EQ(expected_size, input_layer_.capacity());

  SetNumberOfNeuronsInSingleHiddenLayer(layer_size);
  ASSERT_EQ(expected_size, hidden_layer_.capacity());

  SetNumberOfNeuronsInOutputLayer(layer_size);
  ASSERT_EQ(expected_size, output_layer_.capacity());
}

TEST_F(NeuralNetworkBaseTestFixture,
       GivenInvalidInputSize_WhenSettingLayersSize_ThenExceptionsAreThrown) {

  const std::size_t invalid_size{0};

  EXPECT_THROW(SetNumberOfNeuronsInInputLayer(invalid_size),
               std::invalid_argument);

  EXPECT_THROW(SetNumberOfNeuronsInSingleHiddenLayer(invalid_size),
               std::invalid_argument);

  EXPECT_THROW(SetNumberOfNeuronsInOutputLayer(invalid_size),
               std::invalid_argument);
}

TEST_F(
    NeuralNetworkBaseTestFixture,
    GivenValidInputValues_WhenAssigningValuesToInputLayer_ThenCorrectValuesAssigned) {
  SetNumberOfNeuronsInInputLayer(default_input_values.size());

  const std::size_t expected_size{4};

  ASSERT_EQ(expected_size, input_layer_.capacity());

  SetInputValues(default_input_values);

  ASSERT_EQ(expected_size, input_layer_.size());

  for (std::size_t i = 0; i < expected_size; i++) {
    EXPECT_EQ(default_input_values[i], input_layer_[i].GetValue());
  }
}

TEST_F(
    NeuralNetworkBaseTestFixture,
    GivenDefinedLayersSize_WhenReservingSyanpseCapacity_ThenSynapseCapacityIsCorrect) {

  CreateLayersWithDefaultSize();
  ReserveSynapseCapacity();

  const std::size_t expected_synapse_size{30};

  ASSERT_EQ(expected_synapse_size, synapses_.capacity());
}

TEST_F(NeuralNetworkBaseTestFixture,
       GivenDefinedLayersSize_WhenConnectingLayers_ThenLayersAreConnected) {

  CreateLayersWithDefaultSize();
  ReserveSynapseCapacity();

  SetInputValues(default_input_values);
  AssignRandomValuesToLayer(hidden_layer_);

  ConnectLayers(input_layer_, hidden_layer_);

  const std::size_t expected_input_hidden_layer_synapse_size{20};

  ASSERT_EQ(expected_input_hidden_layer_synapse_size, synapses_.size());

  AssignRandomValuesToLayer(output_layer_);
  ConnectLayers(hidden_layer_, output_layer_);

  const std::size_t expected_synapse_size{30};

  ASSERT_EQ(expected_synapse_size, synapses_.size());
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
