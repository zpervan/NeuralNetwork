#include "neural_network_base.h"

#include <algorithm>

namespace Base {

void NeuralNetworkBase::AddInputValues(const std::vector<Value> &input_values) {
  SetNumberOfNeuronsInInputLayer(input_values.size());

  for (auto &value : input_values) {
    input_layer_.emplace_back(Neuron{nullptr, nullptr, value});
  }
}

void NeuralNetworkBase::SetNumberOfNeuronsInInputLayer(size_t size) {
  size > 0 ? input_layer_.reserve(size)
           : throw std::invalid_argument("Size of input values are not valid!");
}

void NeuralNetworkBase::SetNumberOfNeuronsInSingleHiddenLayer(size_t size) {
  hidden_layer_.reserve(size);
}

void NeuralNetworkBase::SetSizeOfOutputLayer(const std::size_t size) {
  output_layer_.reserve(size);
}
void NeuralNetworkBase::ConnectNeurons() {}
} // namespace Base
