#include "../exclusive_or_neural_network.h"
#include "gtest/gtest.h"

class ExclusiveOrNeuralNetworkTestFixture
        : protected ExclusiveOrNeuralNetwork,
          public ::testing::Test {
protected:
    ExclusiveOrNeuralNetworkTestFixture() = default;

    ~ExclusiveOrNeuralNetworkTestFixture()
    {
        input_layer_.clear();
        hidden_layer_.clear();
        output_layer_.clear();
    }

    const NeuralNetworkArchitecture neural_network_architecture_{2, 3, 1,
                                                                 ActivationFunctionType::SIGMOID};
    const std::vector<Value> input_values_{1.0, 0.0};
    const std::vector<Value> output_targets_{0.0};
    const std::size_t expected_synapses_size_{9};

    void CreateDefaultPredefinedConnectedNetwork()
    {

        input_layer_.reserve(2);
        input_layer_.emplace_back(Neuron{0, 1.0});
        input_layer_.emplace_back(Neuron{1, 1.0});

        hidden_layer_.reserve(3);
        hidden_layer_.emplace_back(Neuron{2, 1.0, 0.73});
        hidden_layer_.emplace_back(Neuron{3, 1.3, 0.79});
        hidden_layer_.emplace_back(Neuron{4, 0.8, 0.69});

        output_layer_.reserve(1);
        output_layer_.emplace_back(Neuron{5, 1.2, 0.77, 0.0});

        // Connect first hidden layer neuron with input layer neurons
        synapses_.emplace(
                2, Synapse{&input_layer_.at(0), &hidden_layer_.at(0), 0.8,
                           0});
        synapses_.emplace(
                2, Synapse{&input_layer_.at(1), &hidden_layer_.at(0), 0.2,
                           1});

        // Connect second hidden layer neuron with input layer neurons
        synapses_.emplace(
                3, Synapse{&input_layer_.at(0), &hidden_layer_.at(1), 0.4,
                           3});
        synapses_.emplace(
                3, Synapse{&input_layer_.at(1), &hidden_layer_.at(1), 0.9,
                           4});

        // Connect third hidden layer neuron with input layer neurons
        synapses_.emplace(
                4, Synapse{&input_layer_.at(0), &hidden_layer_.at(2), 0.3,
                           4});
        synapses_.emplace(
                4, Synapse{&input_layer_.at(1), &hidden_layer_.at(2), 0.5,
                           5});

        // Connect output layer neuron with hidden layer neurons
        synapses_.emplace(
                5, Synapse{&hidden_layer_.at(0), &output_layer_.at(0), 0.3,
                           6});
        synapses_.emplace(
                5, Synapse{&hidden_layer_.at(1), &output_layer_.at(0), 0.5,
                           7});
        synapses_.emplace(
                5, Synapse{&hidden_layer_.at(2), &output_layer_.at(0), 0.9,
                           8});

        // ID's aka counts
        synapse_id_ = 8;
        neuron_id_ = 5;

        // Activation function - SIGMOID
        activation_function_type_ = ActivationFunctionType::SIGMOID;
    }

    void SetPredefinedNetworkNeuronValuesToZero()
    {
        hidden_layer_.at(0).value = 0;
        hidden_layer_.at(1).value = 0;
        hidden_layer_.at(2).value = 0;
        output_layer_.at(0).value = 0;
    }

    void CheckSynapsesChildNeuronValuesIsNotZero(
            const std::multimap<Id, Synapse>& actual_values)
    {
        for (const auto& actual_value : actual_values) {
            EXPECT_NE(0, actual_value.second.GetChildNeuron()->value)
                            << "Synapse ID: " << actual_value.first
                            << "\n";
        }
    };
};

TEST_F(ExclusiveOrNeuralNetworkTestFixture,
        GivenDefinedNeuralNetworkArchitecture_WhenApplyingActivationFunctions_ThenCorrectValueReturned)
{
    CreateDefaultPredefinedConnectedNetwork();
    const Value value{hidden_layer_.at(0).value};

    SetActivationFunctionType(ActivationFunctionType::LINEAR);

    Value actual_activation_function_result{
            ApplyActivationFunction(value)};
    Value expected_activation_function_result{1.000};

    EXPECT_NEAR(expected_activation_function_result,
            actual_activation_function_result, 1e-3);

    SetActivationFunctionType(ActivationFunctionType::SIGMOID);

    actual_activation_function_result = ApplyActivationFunction(value);
    expected_activation_function_result = 0.731;

    EXPECT_NEAR(expected_activation_function_result,
            actual_activation_function_result, 1e-3);

    SetActivationFunctionType(ActivationFunctionType::HYPERBOLIC);

    actual_activation_function_result = ApplyActivationFunction(value);
    expected_activation_function_result = 0.103;

    EXPECT_NEAR(expected_activation_function_result,
            actual_activation_function_result, 1e-3);

}

TEST_F(ExclusiveOrNeuralNetworkTestFixture,
        GivenUnknownActivationFunctionType_WhenApplyingActivationFunctionToValue_ThenExceptionIsThrown)
{
    CreateDefaultPredefinedConnectedNetwork();

    SetActivationFunctionType(ActivationFunctionType::UNKNOWN);

    EXPECT_THROW(ApplyActivationFunction(hidden_layer_.at(0).value),
            std::invalid_argument);
}

TEST_F(
        ExclusiveOrNeuralNetworkTestFixture,
        GivenDefinedArchitectureAndInputValues_WhenCalculatingNeuronValues_ThenValuesSuccessfullyAssigned)
{

    DefineNeuralNetworkArchitecture(neural_network_architecture_,
            input_values_, output_targets_);
    ASSERT_EQ(expected_synapses_size_, GetSynapses().size());

    CalculateInitialValues();

    CheckSynapsesChildNeuronValuesIsNotZero(synapses_);
}

TEST_F(
        ExclusiveOrNeuralNetworkTestFixture,
        GivenPredefinedArchitecture_WhenCalculatingInitialValues_ThenExpectedValueIsCalculated)
{

    CreateDefaultPredefinedConnectedNetwork();
    SetPredefinedNetworkNeuronValuesToZero();

    for (std::size_t i = 2; i<=neuron_id_; i++) {
        CalculateNeuronValues(synapses_.equal_range(i));
    }

    const std::array<Value, 4> expected_values{1.0, 1.3, 0.8, 1.235};
    EXPECT_DOUBLE_EQ(expected_values[0], hidden_layer_[0].value);
    EXPECT_DOUBLE_EQ(expected_values[1], hidden_layer_[1].value);
    EXPECT_DOUBLE_EQ(expected_values[2], hidden_layer_[2].value);
    EXPECT_DOUBLE_EQ(expected_values[3], output_layer_[0].value);
}

TEST_F(
        ExclusiveOrNeuralNetworkTestFixture,
        GivenPredefinedArchitecture_WhenApplyingActivationFunction_ThenExpectedValueIsCalculated)
{
    CreateDefaultPredefinedConnectedNetwork();

    const std::array<double, 4> expected_result{0.7310, 0.7858, 0.6899,
                                                0.7685};

    // First hidden neuron and it's children in the output layer
    ApplyActivationFunctionOnNeuronsValue(synapses_.find(2));
    EXPECT_NEAR(expected_result[0],
            hidden_layer_.at(0).activation_func_result, 1e-4);
    // Second hidden neuron and it's children in the output layer
    ApplyActivationFunctionOnNeuronsValue(synapses_.find(3));
    EXPECT_NEAR(expected_result[1],
            hidden_layer_.at(1).activation_func_result, 1e-4);
    // Third hidden neuron and it's children in the output layer
    ApplyActivationFunctionOnNeuronsValue(synapses_.find(4));
    EXPECT_NEAR(expected_result[2],
            hidden_layer_.at(2).activation_func_result, 1e-4);

    ApplyActivationFunctionOnNeuronsValue(synapses_.find(5));
    EXPECT_NEAR(expected_result[3],
            output_layer_.at(0).activation_func_result, 1e-4);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
