#include "neural_network_base.h"

#include <algorithm>

void Base::NeuralNetworkBase::SetSizeOfInputLayer(const std::size_t size) {
  input_layer_.reserve(size);
}

void Base::NeuralNetworkBase::SetSizeOfSingleHiddenLayer(
    const std::size_t size) {
  hidden_layer_.reserve(size);
}

void Base::NeuralNetworkBase::SetSizeOfOutputLayer(const std::size_t size) {
  output_layer_.reserve(size);
}

void Base::NeuralNetworkBase::AddInputValues(const std::vector<Value> &values) {
  if (input_layer_.size() != values.size()) {
    return;
  }

  for(auto& value : values)
  {
    // TODO: Add implementation
  }

}

void Base::NeuralNetworkBase::ConnectNeurons() {}
