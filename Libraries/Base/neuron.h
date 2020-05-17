#ifndef NEURALNETWORK_NEURON_H
#define NEURALNETWORK_NEURON_H

#include "neural_network_architecture_data.h"
#include <utility>
#include <vector>

/// @brief Defines one neuron. Each neuron is aware of it's parent node.
class Neuron {
public:
  Neuron(const Id id, const double value = 0, const double activation_func_result = 0)
      :id_(id), value_(value), activation_func_result_(activation_func_result){};

  Id GetId() const;
  Value GetValue() const;
  double GetActivationFunctionResult() const;

  void SetValue(const double &value);
  void SetActivationFunctionResult(const double &activation_func_result);

  Neuron() = default;
  ~Neuron() = default;

private:
  const Id id_;
  Value value_;
  Value activation_func_result_;
};

#endif // NEURALNETWORK_NEURON_H
