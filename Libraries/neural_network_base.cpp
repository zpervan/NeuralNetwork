#include "neural_network_base.h"

#include <algorithm>

namespace {

template <typename T>
inline bool IsVectorCapacitySameAsVectorSize(std::vector<T> const &vector) {
  return vector.capacity() == vector.size();
}

} // namespace

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

void NeuralNetworkBase::SetInputValues(std::vector<Value> input_values) {

  for (auto &value : input_values) {
    input_layer_.emplace_back(Neuron{value});
  }
}

void NeuralNetworkBase::CreateArchitecture() {

  IsLayersSizeAndCapacitySame();
  ReserveSynapseCapacity();

  AssignRandomValuesToLayer(hidden_layer_);
  AssignRandomValuesToLayer(output_layer_);

  // TODO: Generalize layer connection
  ConnectLayers(input_layer_, hidden_layer_);
  ConnectLayers(hidden_layer_, output_layer_);
}

void NeuralNetworkBase::ConnectLayers(const std::vector<Neuron> &lhs,
                                      const std::vector<Neuron> &rhs) {
  for (std::size_t i = 0; i < lhs.capacity(); i++) {
    for (std::size_t j = 0; j < rhs.capacity(); j++) {
      Synapse synapse;
      synapse.SetParentNeuron(&lhs.at(i));
      synapse.SetChildNeuron(&rhs.at(j));
      synapse.SetWeight(GenerateRandomValue());

      synapses_.emplace_back(synapse);
    }
  }
}

void NeuralNetworkBase::ReserveSynapseCapacity() {
  const std::size_t input_hidden_synapse_size =
      input_layer_.capacity() * hidden_layer_.capacity();

  const std::size_t hidden_output_synapse_size =
      hidden_layer_.capacity() * output_layer_.capacity();

  synapses_.reserve(input_hidden_synapse_size + hidden_output_synapse_size);
}

void NeuralNetworkBase::AssignRandomValuesToLayer(std::vector<Neuron> &inputs) {
  for (std::size_t i = 0; i < inputs.capacity(); i++) {
    inputs.emplace_back(Neuron{GenerateRandomValue()});
  }
}

inline double NeuralNetworkBase::GenerateRandomValue() {
  return distribution_(generator_);
}

void NeuralNetworkBase::IsLayersSizeAndCapacitySame() {
  if (!IsVectorCapacitySameAsVectorSize(input_layer_)) {
    throw std::invalid_argument("Input layer size and capacity do not match!");
  }

  if (!IsVectorCapacitySameAsVectorSize(hidden_layer_)) {
    throw std::invalid_argument("Hidden layer size and capacity do not match!");
  }

  if (!IsVectorCapacitySameAsVectorSize(output_layer_)) {
    throw std::invalid_argument("Output layer size and capacity do not match!");
  }
}
