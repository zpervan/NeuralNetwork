#ifndef NEURALNETWORK_NEURAL_NETWORK_BASE_H
#define NEURALNETWORK_NEURAL_NETWORK_BASE_H

#include <cstddef>
#include <map>
#include <memory>
#include <vector>

// TODO: Better function names, like "SetNumberOfElementsInInputLayers"
// TODO: Create a single function for defining the layers size
namespace Base {

/// @brief Defines one neuron
struct Neuron {
  std::unique_ptr<Neuron> parent;
  std::unique_ptr<Neuron> child;
  double value;
};

using NeuronRelation = std::pair<Neuron, Neuron>;
using Value = double;
using Weight = double;

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

  void AddInputValues(const std::vector<Value>& values);

  void ConnectNeurons();

protected:
  std::vector<Neuron> input_layer_{};
  std::vector<Neuron> hidden_layer_{};
  std::vector<Neuron> output_layer_{};
  std::vector<Weight> weights_{};
  std::map<NeuronRelation, Weight> neuron_relations_;
};

} // namespace Base
#endif // NEURALNETWORK_NEURAL_NETWORK_BASE_H
