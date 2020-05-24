#include "exclusive_or_neural_network.h"

#include <algorithm>
#include <iostream>

void ExclusiveOrNeuralNetwork::DefineNeuralNetworkArchitecture(
		NeuralNetworkArchitecture network,
		const std::vector<Value>& input_values,
		const std::vector<Value>& output_targets)
{
	SetNumberOfNeuronsInInputLayer(network.input_layer_size);
	SetNumberOfNeuronsInSingleHiddenLayer(
			network.single_hidden_layer_size);
	SetNumberOfNeuronsInOutputLayer(network.output_layer_layer_size);
	CreateNetwork(input_values, output_targets);
	SetActivationFunctionType(network.activation_function_type);
}

void ExclusiveOrNeuralNetwork::CalculateInitialValues()
{
	for (std::size_t i = 0; i<neuron_id_; i++) {

		std::pair<SynapseIterator, SynapseIterator> found_synapses =
				synapses_.equal_range(i);

		CalculateNeuronValues(found_synapses);
		ApplyActivationFunctionOnNeuronsValue(found_synapses.first);
	}
}

void ExclusiveOrNeuralNetwork::PrintNeuralNetworkData()
{
	std::cout << "\e[30m- - - - Input layer - - - -\e[0m \n";
	PrintLayerData(input_layer_);
	std::cout << "\e[92m- - - - Hidden layer - - - -\e[0m \n";
	PrintLayerData(hidden_layer_);
	std::cout << "\e[33m- - - - Output layer - - - -\e[0m \n";
	PrintLayerData(output_layer_);
}

void ExclusiveOrNeuralNetwork::PrintLayerData(
		const std::vector<Neuron>& layer) const
{
	for (auto neuron : layer) {
		std::cout << "ID: " << neuron.id << "\n";
		std::cout << "Value: " << neuron.value << "\n";
		std::cout << "Activation function result: "
				  << neuron.activation_func_result << "\n";
	}
}

void ExclusiveOrNeuralNetwork::CalculateNeuronValues(
		const std::pair<SynapseIterator, SynapseIterator>& found_synapses)
{

	for (auto it = found_synapses.first; it!=found_synapses.second; it++) {
		Value child_neuron_value = it->second.GetChildNeuron()->value;
		/// @TODO: Consider the situation when the activation function result is 0 - how to handle this situation?
		if (it->second.GetParentNeuron()->activation_func_result
				>0) {
			child_neuron_value +=
					it->second.GetParentNeuron()->activation_func_result
							*it->second.GetWeight();
		}
		else {
			// Because input layer does not have a activation function result, take the input value and calculate the
			// child neuron value
			child_neuron_value += it->second.GetParentNeuron()->value
					*it->second.GetWeight();
		}
		it->second.GetChildNeuron()->value = child_neuron_value;
	}
}

void ExclusiveOrNeuralNetwork::ApplyActivationFunctionOnNeuronsValue(
		const SynapseIterator& calculated_value)
{
	auto neuron_value = calculated_value->second.GetChildNeuron()->value;
	calculated_value->second.GetChildNeuron()->activation_func_result =
			ApplyActivationFunction(neuron_value);
}

double ExclusiveOrNeuralNetwork::ApplyActivationFunction(
		const double value)
{
	switch (activation_function_type_) {
	case ActivationFunctionType::LINEAR :
		return ActivationFunction::Linear(value);
	case ActivationFunctionType::SIGMOID :
		return ActivationFunction::Sigmoid(value);
	case ActivationFunctionType::HYPERBOLIC:
		return ActivationFunction::HyperbolicTangent(value);
	default:
		throw std::invalid_argument("No activation function is defined!");
	}
}

void ExclusiveOrNeuralNetwork::SetActivationFunctionType(
		ActivationFunctionType activation_function_type)
{
	activation_function_type_ = activation_function_type;
}
