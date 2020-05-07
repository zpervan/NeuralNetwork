#include "feed_forward_network.h"

#include <algorithm>
#include <iostream>

using SynapseIterator = std::multimap<Id, Synapse>::iterator;

void FeedForwardNetwork::DefineNeuralNetworkArchitecture(
    NeuralNetworkArchitecture network, const std::vector<Value> &input_values) {

  SetNumberOfNeuronsInInputLayer(network.input_layer_size);
  SetNumberOfNeuronsInSingleHiddenLayer(network.single_hidden_layer_size);
  SetNumberOfNeuronsInOutputLayer(network.output_layer_layer_size);
  CreateNetwork(input_values);
}

// TODO: Refactor. Please and thank you.
void FeedForwardNetwork::TrainOnce() {

  for (std::size_t i = 0; i < neuron_id_; i++) {
    std::pair<SynapseIterator, SynapseIterator> found_synapses =
        synapses_.equal_range(i);

    for (auto it = found_synapses.first; it != found_synapses.second; it++) {
      std::cout << "Synapse ID: " << it->second.GetId() << "\n";
      std::cout << "Synapse random weight: " << it->second.GetWeight() << "\n";
      std::cout << "Child neuron value before calculation: "
                << it->second.GetChildNeuron()->GetValue() << "\n";

      Value child_neuron_value = it->second.GetChildNeuron()->GetValue();

      child_neuron_value +=
          it->second.GetParentNeuron()->GetValue() * it->second.GetWeight();

      it->second.GetChildNeuron()->SetValue(child_neuron_value);

      std::cout << "Child neuron value after calculation: "
                << it->second.GetChildNeuron()->GetValue() << "\n";
    }
  }
}
