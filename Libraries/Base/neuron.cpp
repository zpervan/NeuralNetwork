#include "neuron.h"

Id Neuron::GetId() const { return id_; }

Value Neuron::GetValue() const { return value_; }

Value Neuron::GetActivationFunctionResult() const { return activation_func_result_; }

Value Neuron::GetOutputTarget() const { return output_target_; }

void Neuron::SetValue(Value value) { value_ = value; }

void Neuron::SetActivationFunctionResult(
        Value activation_func_result) { activation_func_result_ = activation_func_result; }

void Neuron::SetOutputTarget(
        const Value output_target) { output_target_ = output_target; }
