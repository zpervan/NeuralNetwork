#ifndef NEURALNETWORK_SYNAPSE_H
#define NEURALNETWORK_SYNAPSE_H

#include "neuron.h"

using Weight = double;

/// @brief Defines a connection between two neurons - synapse. Each synapse has
/// a weight which should be a random value while instantiating. Afterwards it's
/// calculated during training.
class Synapse {
public:
  Synapse(const Neuron *parent, const Neuron *child, double weight);
  Synapse() = default;
  ~Synapse();

  void SetParentNeuron(const Neuron *parent);
  void SetChildNeuron(const Neuron *child);
  void SetWeight(const Weight weight);

  const Neuron *GetParentNeuron() const;
  const Neuron *GetChildNeuron() const;
  Weight GetWeight() const;

private:
  const Neuron *parent_{nullptr};
  const Neuron *child_{nullptr};
  double weight_{0.0};

public:
  double getWeight() const;
};

#endif // NEURALNETWORK_SYNAPSE_H
