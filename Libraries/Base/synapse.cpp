#include "synapse.h"

Synapse::Synapse(const Neuron *parent, const Neuron *child, double weight)
    : parent_(parent), child_(child), weight_(weight) {}

Synapse::~Synapse() {
  weight_ = 0;
  parent_ = nullptr;
  child_ = nullptr;
}

void Synapse::SetParentNeuron(const Neuron *parent) {
  if (parent_ == nullptr) {
    parent_ = parent;
  }
}

void Synapse::SetChildNeuron(const Neuron *child) {
  if (child_ == nullptr) {
    child_ = child;
  }
}

void Synapse::SetWeight(const Weight weight) { weight_ = weight; }

const Neuron *Synapse::GetParentNeuron() const { return parent_; }

const Neuron *Synapse::GetChildNeuron() const { return child_; }

Weight Synapse::GetWeight() const { return weight_; }

