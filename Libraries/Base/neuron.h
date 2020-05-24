#ifndef NEURALNETWORK_NEURON_H
#define NEURALNETWORK_NEURON_H

#include "neural_network_architecture_data.h"
#include <utility>
#include <vector>
/// @todo: consider making it a struct
/// @brief Defines one neuron. Each neuron is aware of it's parent node.
class Neuron {
public:
    Neuron(const Id id, const Value value = 0,
            const Value activation_func_result = 0,
            const Value output_target = 0)
            :id_(id), value_(value),
             activation_func_result_(activation_func_result),
             output_target_(output_target) { };

    Id GetId() const;
    Value GetValue() const;
    Value GetActivationFunctionResult() const;
    Value GetOutputTarget() const;

    void SetValue(Value value);
    void SetActivationFunctionResult(Value activation_func_result);
    void SetOutputTarget(Value output_target);

private:
    const Id id_{0};
    Value value_{0.0};
    Value activation_func_result_{0.0};
    Value output_target_{0.0};
};

#endif // NEURALNETWORK_NEURON_H
