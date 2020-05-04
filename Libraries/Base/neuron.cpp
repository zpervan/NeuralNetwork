#include "neuron.h"

double Neuron::GetValue() const { return value_; }

Id Neuron::GetId() const { return id_; }

void Neuron::SetValue(const double &value) { value_ = value; }
