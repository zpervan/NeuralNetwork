#include "neural_network_base.h"

void NeuralNetworkBase::SetSizeOfInputLayer(const std::size_t size) {
  input_layer.reserve(size);
}

void NeuralNetworkBase::SetSizeOfSingleHiddenLayer(const std::size_t size) {
  hidden_layer.reserve(size);
}

void NeuralNetworkBase::SetSizeOfOutputLayer(const std::size_t size) {
  output_layer.reserve(size);
}
