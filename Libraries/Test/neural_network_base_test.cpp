#include "../neural_network_base.h"

#include <memory>

#include "gtest/gtest.h"

class NeuralNetworkBaseTestFixture : protected NeuralNetworkBase,
                                     public ::testing::Test {
protected:
  const std::vector<Value> default_input_values{3.0, 2.0, 1.0, 0.0};
  const std::array<std::pair<Id, Value>, 2> default_values{
      {{0, 1.0}, {1, 2.0}}};

  void CreateLayersWithDefaultSize() {
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

  Neuron *parent_neuron{
      new Neuron{default_values[0].first, default_values[0].second}};
  Neuron *child_neuron{
      new Neuron{default_values[1].first, default_values[1].second}};

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

  const std::array<Neuron *, 2> neurons{
      new Neuron{default_values[0].first, default_values[0].second},
      new Neuron{default_values[1].first, default_values[1].second}};

  auto *synapse = new Synapse(neurons[0], neurons[1], 0.0);

  ASSERT_TRUE(synapse);

  EXPECT_EQ(default_values[0].second, synapse->GetParentNeuron()->GetValue());
  EXPECT_EQ(default_values[1].second, synapse->GetChildNeuron()->GetValue());

  const std::array<Neuron *, 2> new_neurons{new Neuron(0, 3.0),
                                            new Neuron(1, 4.0)};

  synapse->SetParentNeuron(new_neurons[0]);
  synapse->SetChildNeuron(new_neurons[1]);

  EXPECT_EQ(default_values[0].second, synapse->GetParentNeuron()->GetValue());
  EXPECT_EQ(default_values[1].second, synapse->GetChildNeuron()->GetValue());
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

  const std::size_t actual_synapse_capacity =
      (input_layer_.capacity() * hidden_layer_.capacity()) +
      (hidden_layer_.capacity() * output_layer_.capacity());

  const std::size_t expected_synapse_capacity{30};

  ASSERT_EQ(expected_synapse_capacity, actual_synapse_capacity);
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

// @TODO: Adapt test to use map function calls
TEST_F(
    NeuralNetworkBaseTestFixture,
    GivenInputValues_WhenArchitectureIsDefined_ThenNetworkIsSucessfulyCreated) {
  const std::size_t neurons_in_input_layer{2};
  SetNumberOfNeuronsInInputLayer(neurons_in_input_layer);

  const std::size_t neurons_in_hidden_layer{3};
  SetNumberOfNeuronsInSingleHiddenLayer(neurons_in_hidden_layer);

  const std::size_t neurons_in_output_layer{1};
  SetNumberOfNeuronsInOutputLayer(neurons_in_output_layer);

  const std::vector<Value> input_values{1.0, 0.0};
  CreateNetwork(input_values);

  const std::size_t expected_synapses_size{9};
  ASSERT_EQ(expected_synapses_size, synapses_.size());

  const std::array<Value, 2> expected_values{1.0, 0.0};

  // Values in the hidden and output layer are random so we compare that they
  // are not equal to zero (invalid value)
  const Value invalid_value{0.0};
  const Weight invalid_weight{0.0};
//
//  EXPECT_EQ(expected_values[0], synapses_[1].GetParentNeuron()->GetValue());
//  EXPECT_NE(invalid_value, synapses_[1].GetChildNeuron()->GetValue());
//
//  EXPECT_EQ(expected_values[1], synapses_[3].GetParentNeuron()->GetValue());
//  EXPECT_NE(invalid_value, synapses_[3].GetChildNeuron()->GetValue());
//
//  EXPECT_EQ(expected_values[1], synapses_[5].GetParentNeuron()->GetValue());
//  EXPECT_NE(invalid_value, synapses_[5].GetChildNeuron()->GetValue());
//
//  EXPECT_NE(invalid_value, synapses_[7].GetParentNeuron()->GetValue());
//  EXPECT_NE(invalid_value, synapses_[7].GetChildNeuron()->GetValue());
//
//  EXPECT_NE(invalid_weight, synapses_[3].GetWeight());
//  EXPECT_NE(invalid_weight, synapses_[7].GetWeight());
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
