#ifndef NEURALNETWORK_NEURON_H
#define NEURALNETWORK_NEURON_H

#include "neural_network_architecture_data.h"

/// @brief Defines one neuron
class Neuron {
public:
  explicit Neuron(const Id id, const double value) : id_(id), value_(value){};

  Id GetId() const;
  double GetValue() const;

  void SetValue(const double &value);

  Neuron() = default;
  ~Neuron() = default;

private:
  const Id id_;
  double value_;
};

#endif // NEURALNETWORK_NEURON_H
