#ifndef NEURALNETWORK_SIMPLE_NEURAL_NETWORK_H
#define NEURALNETWORK_SIMPLE_NEURAL_NETWORK_H

#include "Base/neural_network_architecture_data.h"
#include "neural_network_base.h"

using SynapseIterator = std::multimap<Id, Synapse>::iterator;

/// @todo: Add activation function
class ExclusiveOrNeuralNetwork : public NeuralNetworkBase {
public:
  /// @brief Define a neural network architecture with given layer sizes and
  /// input values.
  /// @param network Defines the size of the network layers
  /// @param input_values Values which will be assigned to the input layer
  /// neurons
  void DefineNeuralNetworkArchitecture(NeuralNetworkArchitecture network,
                                       const std::vector<Value> &input_values);

  /// @brief Calculates the initial values for defined neural network
  /// architecture
  void CalculateInitialValues();

protected:
  void CalculateChildNeuronValue(
      const std::pair<SynapseIterator, SynapseIterator> &found_synapses) const;
};

#endif // NEURALNETWORK_SIMPLE_NEURAL_NETWORK_H
