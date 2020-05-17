#include "exclusive_or_neural_network.h"

#include <algorithm>
#include <iostream>

void ExclusiveOrNeuralNetwork::DefineNeuralNetworkArchitecture(
        NeuralNetworkArchitecture network, const std::vector<Value>& input_values)
{

    SetNumberOfNeuronsInInputLayer(network.input_layer_size);
    SetNumberOfNeuronsInSingleHiddenLayer(network.single_hidden_layer_size);
    SetNumberOfNeuronsInOutputLayer(network.output_layer_layer_size);
    CreateNetwork(input_values);
}

void ExclusiveOrNeuralNetwork::CalculateInitialValues()
{
    for (std::size_t i = 0; i<=neuron_id_; i++) {

        std::pair<SynapseIterator, SynapseIterator> found_synapses =
                synapses_.equal_range(i);

        CalculateNeuronValues(found_synapses);
        ApplyActivationFunctionToNeuronValues(found_synapses);
    }
    std::cout << "Synapse size (num of elements):  " << synapses_.size() << "\n";

    for (std::size_t i = i; i<=synapse_id_; i++) {
        std::pair<SynapseIterator, SynapseIterator> found_synapses =
                synapses_.equal_range(i);

    }
}

void ExclusiveOrNeuralNetwork::CalculateNeuronValues(
        const std::pair<SynapseIterator, SynapseIterator>& found_synapses)
{

    for (auto it = found_synapses.first; it!=found_synapses.second; it++) {
        Value child_neuron_value = it->second.GetChildNeuron()->GetValue();

        if (it->second.GetParentNeuron()->GetActivationFunctionResult()>0) {
            child_neuron_value += it->second.GetParentNeuron()->GetActivationFunctionResult()*it->second.GetWeight();
        }
        else {
            // Because input layer does not have a activation function result, take the input value and calculate the
            // child neuron value
            child_neuron_value += it->second.GetParentNeuron()->GetValue()*it->second.GetWeight();
        }
        it->second.GetChildNeuron()->SetValue(child_neuron_value);
    }
}

/// @todo: Add activation function type
void ExclusiveOrNeuralNetwork::ApplyActivationFunctionToNeuronValues(
        const std::pair<SynapseIterator, SynapseIterator>& calculated_values)
{

    for (auto it = calculated_values.first; it!=calculated_values.second;
         it++) {
        Value child_neuron_value = it->second.GetChildNeuron()->GetValue();

        it->second.GetChildNeuron()->SetActivationFunctionResult(
                ActivationFunction::Sigmoid(child_neuron_value));
    }
}
