#ifndef NEURALNETWORK_NEURAL_NETWORK_BASE_H
#define NEURALNETWORK_NEURAL_NETWORK_BASE_H

#include "Base/neuron.h"
#include "Base/synapse.h"

#include <cstddef>
#include <memory>
#include <random>
#include <vector>

// TODO: Create a single function for defining the layers size

/// @brief Base class for defining the neural network architecture
class NeuralNetworkBase {
public:
  /// @brief Defines the numbers of neurons in the input layer
  /// @param size Value representing how many neurons will the input layer have
  void SetNumberOfNeuronsInInputLayer(std::size_t size);

  /// @brief Defines the numbers of neurons in the single hidden layer
  /// @param size Value representing how many neurons will the hidden layer have
  void SetNumberOfNeuronsInSingleHiddenLayer(std::size_t size);

  /// @brief Defines the numbers of neurons in the output layer
  /// @param size Value representing how many neurons will the output layer have
  void SetNumberOfNeuronsInOutputLayer(std::size_t size);

  /// @brief
  void SetInputValues(std::vector<Value> input_values);

  /// @brief
  void CreateArchitecture();

protected:
  /// @brief
  void ReserveSynapseCapacity();

  /// @brief
  /// @param input_values
  void AssignRandomValuesToLayer(std::vector<Neuron> &inputs);

  /// @brief
  /// @param lhs
  /// @param rhs
  void ConnectLayers(const std::vector<Neuron> &lhs,
                     const std::vector<Neuron> &rhs);

  /// @brief
  inline double GenerateRandomValue();

  /// @brief
  void IsLayersSizeAndCapacitySame();
  std::random_device random_device_;
  std::default_random_engine generator_{random_device_()};

  std::uniform_real_distribution<double> distribution_{0.0, 1.0};
  std::vector<Neuron> input_layer_{};
  std::vector<Neuron> hidden_layer_{};
  std::vector<Neuron> output_layer_{};
  // TODO: Consider using dequeue for synapses because back propagation will be
  // introduced
  std::vector<Synapse> synapses_{};
};

#endif // NEURALNETWORK_NEURAL_NETWORK_BASE_H
