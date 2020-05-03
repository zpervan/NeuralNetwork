#include "neural_network_base.h"

#include <algorithm>

void NeuralNetworkBase::AssignInputValues(
    const std::vector<Value> &input_values) {

  SetNumberOfNeuronsInInputLayer(input_values.size());

  for (auto &value : input_values) {
    input_layer_.emplace_back(Neuron(value));
  }
}

void NeuralNetworkBase::SetNumberOfNeuronsInInputLayer(size_t size) {
  size > 0 ? input_layer_.reserve(size)
           : throw std::invalid_argument("Size of the input values is 0!");
}

void NeuralNetworkBase::SetNumberOfNeuronsInSingleHiddenLayer(size_t size) {
  (size < input_layer_.capacity()) || (size == 0)
      ? throw std::invalid_argument("Size of hidden layer should be at least "
                                    "the size of the input layer!")
      : hidden_layer_.reserve(size);
}

void NeuralNetworkBase::SetNumberOfNeuronsInOutputLayer(size_t size) {
  size > 0 ? output_layer_.reserve(size)
           : throw std::invalid_argument("Size of output layer is not valid!");
}

void NeuralNetworkBase::ConnectNeurons() {}

