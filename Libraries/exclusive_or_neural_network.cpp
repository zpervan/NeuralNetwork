#include "exclusive_or_neural_network.h"

#include <algorithm>
#include <iostream>

void ExclusiveOrNeuralNetwork::DefineNeuralNetworkArchitecture(
    NeuralNetworkArchitecture network, const std::vector<Value> &input_values) {

  SetNumberOfNeuronsInInputLayer(network.input_layer_size);
  SetNumberOfNeuronsInSingleHiddenLayer(network.single_hidden_layer_size);
  SetNumberOfNeuronsInOutputLayer(network.output_layer_layer_size);
  CreateNetwork(input_values);
}

void ExclusiveOrNeuralNetwork::CalculateInitialValues() {

  for (std::size_t i = 0; i < neuron_id_; i++) {

    std::pair<SynapseIterator, SynapseIterator> found_synapses =
        synapses_.equal_range(i);

    CalculateChildNeuronValue(found_synapses);
  }
}
void ExclusiveOrNeuralNetwork::CalculateChildNeuronValue(
    const std::pair<SynapseIterator, SynapseIterator> &found_synapses) const {

  for (auto it = found_synapses.first; it != found_synapses.second; it++) {
    Value child_neuron_value = it->second.GetChildNeuron()->GetValue();

    child_neuron_value +=
        it->second.GetParentNeuron()->GetValue() * it->second.GetWeight();

    it->second.GetChildNeuron()->SetValue(child_neuron_value);
  }
}
