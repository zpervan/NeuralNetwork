#ifndef NEURALNETWORK_NEURAL_NETWORK_ARCHITECTURE_DATA_H
#define NEURALNETWORK_NEURAL_NETWORK_ARCHITECTURE_DATA_H

#include <cstddef>
#include <cstdint>


using Id = std::uint64_t;
using Value = double;
using Weight = double;

// TODO: Add multiple levels of hidden layer
struct NeuralNetworkArchitecture {
  std::size_t input_layer_size;
  std::size_t single_hidden_layer_size;
  std::size_t output_layer_layer_size;
};

#endif // NEURALNETWORK_NEURAL_NETWORK_ARCHITECTURE_DATA_H
