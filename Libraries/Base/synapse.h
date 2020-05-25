#ifndef NEURALNETWORK_SYNAPSE_H
#define NEURALNETWORK_SYNAPSE_H

#include "neural_network_data.h"

/// @brief Defines a connection between two neurons - synapse.
class Synapse {
public:
  Synapse(Neuron *parent, Neuron *child, Weight weight, Id id = 0);
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
  Weight weight_{0.0};
  Id id_{0};
};

#endif // NEURALNETWORK_SYNAPSE_H
