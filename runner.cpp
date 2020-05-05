#include "Libraries/feed_forward_network.h"

int main() {
  NeuralNetworkArchitecture network_architecture{2, 3, 1};
  std::vector<Value> input_values{1, 1};

  FeedForwardNetwork feed_forward_network;
  feed_forward_network.DefineNeuralNetworkArchitecture(network_architecture,
                                                       input_values);
  feed_forward_network.TrainOnce();
  return 0;
}
