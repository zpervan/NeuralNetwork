project(Base)

add_library(NeuralNetworkComponents
        STATIC
        neuron.h
        neuron.cpp
        synapse.h
        synapse.cpp
        neural_network_architecture_data.h
        )
