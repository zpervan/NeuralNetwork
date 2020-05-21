#include "Libraries/exclusive_or_neural_network.h"

int main()
{
    NeuralNetworkArchitecture network_architecture{2, 3, 1, ActivationFunctionType::SIGMOID};

    ExclusiveOrNeuralNetwork or_nn;
    or_nn.DefineNeuralNetworkArchitecture(network_architecture, {1, 1});
    or_nn.CalculateInitialValues();
    or_nn.PrintNeuralNetworkData();

    return 0;
}
