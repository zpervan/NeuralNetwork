#ifndef NEURALNETWORK_ACTIVATION_FUNCTIONS_H
#define NEURALNETWORK_ACTIVATION_FUNCTIONS_H

#include <cmath>

enum class ActivationFunctionType { LINEAR = 0, SIGMOID = 1, HYPERBOLIC = 2 };

namespace ActivationFunction {

static inline double Linear(const double x) { return x; }

static inline double Sigmoid(const double x) { return 1 / (1 + std::exp(-x)); }

static inline double HyperbolicTangent(const double x) {
  return (1 - std::exp(-2 * x)) / (1 + std::exp(2 * x));
}

} // namespace ActivationFunction

#endif // NEURALNETWORK_ACTIVATION_FUNCTIONS_H
