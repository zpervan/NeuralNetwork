#include "feed_forward_network.h"
#include <iostream>

void FeedForwardNetwork::DefineNeuralNetworkArchitecture(
    NeuralNetworkArchitecture network, const std::vector<Value> &input_values) {

  SetNumberOfNeuronsInInputLayer(network.input_layer_size);
  SetNumberOfNeuronsInSingleHiddenLayer(network.single_hidden_layer_size);
  SetNumberOfNeuronsInOutputLayer(network.output_layer_layer_size);
  CreateNetwork(input_values);
}

void FeedForwardNetwork::TrainOnce() {
  for (std::size_t i = 1; i < neuron_id_; i++) {
    auto element_pair = synapses_.equal_range(i);
    std::cout << "First it elem: " << element_pair.first->second.GetParentNeuron()->GetId() << std::endl;
    std::cout << "Second it elem: " << element_pair.second->second.GetParentNeuron()->GetId() << std::endl;
  }
}
