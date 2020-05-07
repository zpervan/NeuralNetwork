#include "Libraries/exclusive_or_neural_network.h"

int main() {
  NeuralNetworkArchitecture network_architecture{2, 3, 1};
  std::vector<Value> input_values{1, 1};

  ExclusiveOrNeuralNetwork exclusive_or_neural_network;
  exclusive_or_neural_network.DefineNeuralNetworkArchitecture(network_architecture,
                                                       input_values);
  exclusive_or_neural_network.CalculateInitialValues();
  return 0;
}
