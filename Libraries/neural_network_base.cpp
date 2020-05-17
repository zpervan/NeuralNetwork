#include "neural_network_base.h"

#include <algorithm>
#include <exception>
#include <iostream>

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

void NeuralNetworkBase::SetInputValues(const std::vector<Value> &input_values) {
  for (auto &value : input_values) {
    input_layer_.emplace_back(Neuron{neuron_id_, value, value});
    neuron_id_++;
  }
}

void NeuralNetworkBase::CreateNetwork(const std::vector<double> &input_values) {
  CreateLayers(input_values);
  ConnectNetwork();
}

void NeuralNetworkBase::CreateLayers(const std::vector<double> &input_values) {
  SetInputValues(input_values);
  AddNeuronToLayer(hidden_layer_);
  AddNeuronToLayer(output_layer_);
  AreLayersSizeAndCapacitySame();
}

void NeuralNetworkBase::ConnectNetwork() {
  ConnectLayers(input_layer_, hidden_layer_);
  ConnectLayers(hidden_layer_, output_layer_);
}

void NeuralNetworkBase::ConnectLayers(std::vector<Neuron> &parent,
                                      std::vector<Neuron> &child) {
  for (std::size_t i = 0; i < parent.capacity(); i++) {
    for (std::size_t j = 0; j < child.capacity(); j++) {
      Synapse synapse;
      synapse.SetParentNeuron(&parent.at(i));
      synapse.SetChildNeuron(&child.at(j));
      synapse.SetWeight(GenerateRandomValue());
      synapse.SetId(synapse_id_);
      synapse_id_++;

      synapses_.emplace(synapse.GetChildNeuron()->GetId(), synapse);
    }
  }
}

void NeuralNetworkBase::AddNeuronToLayer(std::vector<Neuron> &layer) {
  for (std::size_t i = 0; i < layer.capacity(); i++) {
    layer.emplace_back(Neuron{neuron_id_, 0});
    neuron_id_++;
  }
}

inline double NeuralNetworkBase::GenerateRandomValue() {
  return distribution_(generator_);
}

void NeuralNetworkBase::AreLayersSizeAndCapacitySame() {
  if (!IsVectorCapacitySameAsVectorSize(input_layer_)) {
    throw std::invalid_argument(
        "Input layer capacity and elements size do not match!");
  }

  if (!IsVectorCapacitySameAsVectorSize(hidden_layer_)) {
    throw std::invalid_argument(
        "Hidden layer capacity and elements size do not match!");
  }

  if (!IsVectorCapacitySameAsVectorSize(output_layer_)) {
    throw std::invalid_argument(
        "Output layer capacity and elements size do not match!");
  }
}

std::multimap<Id, Synapse> NeuralNetworkBase::GetSynapses() const {
  if (synapses_.empty()) {
    std::cerr << "Synapses are not yet defined!\n";
    return {};
  }

  return synapses_;
}
