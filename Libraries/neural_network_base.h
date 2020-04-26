#ifndef NEURALNETWORK_NEURAL_NETWORK_BASE_H
#define NEURALNETWORK_NEURAL_NETWORK_BASE_H

#include <cstddef>
#include <vector>

// TODO: Better function names, like "SetNumberOfElementsInInputLayers"
// TODO: Create a single function for defining the layers size

/// @brief Base class for defining the neural network architecture
class NeuralNetworkBase {
public:
  /// @brief Defines the numbers of neurons in the input layer
  /// @param size Value representing how many neurons will the input layer have
  void SetSizeOfInputLayer(std::size_t size);

  /// @brief Defines the numbers of neurons in the single hidden layer
  /// @param size Value representing how many neurons will the hidden layer have
  void SetSizeOfSingleHiddenLayer(std::size_t size);

  /// @brief Defines the numbers of neurons in the output layer
  /// @param size Value representing how many neurons will the output layer have
  void SetSizeOfOutputLayer(std::size_t size);

protected:
  std::vector<double> input_layer{};
  std::vector<double> hidden_layer{};
  std::vector<double> output_layer{};
};

#endif // NEURALNETWORK_NEURAL_NETWORK_BASE_H
