#include "neuron.h"

double Neuron::GetValue() const { return value_; }

Id Neuron::GetId() const { return id_; }

void Neuron::SetValue(const double &value) { value_ = value; }

double Neuron::GetActivationFunctionResult() const { return activation_func_result_; }

void Neuron::SetActivationFunctionResult(const double &result) { activation_func_result_ = result; }
