#include "synapse.h"

Synapse::Synapse(Neuron *parent, Neuron *child, double weight, const Id id)
    : parent_(parent), child_(child), weight_(weight), id_(id) {}

Synapse::~Synapse() {
  weight_ = 0;
  parent_ = nullptr;
  child_ = nullptr;
}

void Synapse::SetParentNeuron(Neuron *parent) {
  if (parent_ == nullptr) {
    parent_ = parent;
  }
}

void Synapse::SetChildNeuron(Neuron *child) {
  if (child_ == nullptr) {
    child_ = child;
  }
}

void Synapse::SetWeight(const Weight weight) { weight_ = weight; }

void Synapse::SetId(const Id id) { id_ = id; }

Neuron *Synapse::GetParentNeuron() const { return parent_; }

Neuron *Synapse::GetChildNeuron() const { return child_; }

Weight Synapse::GetWeight() const { return weight_; }

Id Synapse::GetId() const { return id_; }
