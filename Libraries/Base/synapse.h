#ifndef NEURALNETWORK_SYNAPSE_H
#define NEURALNETWORK_SYNAPSE_H

#include "neuron.h"

/// @brief Defines a connection between two neurons - synapse. Each synapse has
/// a weight which should be a random value while instantiating. Afterwards it's
/// calculated during training.
/// @todo: Adjust tests to check Id
class Synapse {
public:
  Synapse(Neuron *parent, Neuron *child, double weight, Id id = 0);
  Synapse() = default;
  ~Synapse();

  void SetParentNeuron(Neuron *parent);
  void SetChildNeuron(Neuron *child);
  void SetWeight(Weight weight);
  void SetId(Id id);

  Neuron *GetParentNeuron() const;
  Neuron *GetChildNeuron() const;
  Weight GetWeight() const;
  Id GetId() const;

private:
  Neuron *parent_{nullptr};
  Neuron *child_{nullptr};
  double weight_{0.0};
  Id id_{0};
};

#endif // NEURALNETWORK_SYNAPSE_H
