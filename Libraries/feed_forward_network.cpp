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



   std::cout << "Hello World!\n";
}
