#ifndef NEURALNETWORK_NEURAL_NETWORK_BASE_H
#define NEURALNETWORK_NEURAL_NETWORK_BASE_H

#include "Base/neuron.h"
#include "Base/synapse.h"

#include <cstddef>
#include <map>
#include <memory>
#include <vector>

// TODO: Better function names, like "SetNumberOfElementsInInputLayers"
// TODO: Create a single function for defining the layers size
namespace Base {

/// @brief Base class for defining the neural network architecture
class NeuralNetworkBase {
public:
  /// @brief
  /// @param input_values
  void AssignInputValues(const std::vector<Value> &input_values);

  /// @brief Defines the numbers of neurons in the single hidden layer
  /// @param size Value representing how many neurons will the hidden layer have
  void SetNumberOfNeuronsInSingleHiddenLayer(std::size_t size);

  /// @brief Defines the numbers of neurons in the output layer
  /// @param size Value representing how many neurons will the output layer have
  void SetNumberOfNeuronsInOutputLayer(std::size_t size);

  /// @brief
  void ConnectNeurons();

protected:
  /// @brief Defines the numbers of neurons in the input layer
  /// @param size Value representing how many neurons will the input layer have
  void SetNumberOfNeuronsInInputLayer(std::size_t size);

  std::vector<Neuron> input_layer_{};
  std::vector<Neuron> hidden_layer_{};
  std::vector<Neuron> output_layer_{};
};

} // namespace Base
#endif // NEURALNETWORK_NEURAL_NETWORK_BASE_H
