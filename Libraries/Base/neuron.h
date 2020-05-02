#ifndef NEURALNETWORK_NEURON_H
#define NEURALNETWORK_NEURON_H

using Value = double;

/// @brief Defines one neuron
class Neuron {
public:
  explicit Neuron(const double value) : value_(value){};
  double GetValue() const;
  void SetValue(const double &value);

  Neuron() = default;
  Neuron(Neuron &neuron) = delete;
  Neuron(Neuron &&neuron) = default;
  Neuron &operator=(const Neuron &neuron) = delete;
  Neuron &operator=(Neuron &&neuron) = default;
  ~Neuron() = default;

private:
  double value_;
};

#endif // NEURALNETWORK_NEURON_H
