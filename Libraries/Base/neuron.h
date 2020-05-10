#ifndef NEURALNETWORK_NEURON_H
#define NEURALNETWORK_NEURON_H

#include "neural_network_architecture_data.h"
#include <utility>
#include <vector>

/// @brief Defines one neuron. Each neuron is aware of it's parent node.
class Neuron {
public:
  Neuron(const Id id, const double value = 0, const double result = 0)
      : id_(id), value_(value), result_(result){};

  Id GetId() const;
  Value GetValue() const;
  double GetResult() const;

  void SetValue(const double &value);
  void SetResult(const double &result);

  Neuron() = default;
  ~Neuron() = default;

private:
  const Id id_;
  Value value_;
  Value result_;
};

#endif // NEURALNETWORK_NEURON_H
