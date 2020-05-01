#ifndef NEURALNETWORK_SYNAPSE_H
#define NEURALNETWORK_SYNAPSE_H

#include "neuron.h"

class Synapse{

protected:
  Neuron* parent_;
  Neuron* child_;
  double weight_;
};

#endif // NEURALNETWORK_SYNAPSE_H
