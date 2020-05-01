#include "../neural_network_base.h"

#include <memory>

#include "gtest/gtest.h"
namespace Base {
class NeuralNetworkBaseTestFixture : protected Base::NeuralNetworkBase,
                                     public ::testing::Test {
protected:
  const std::vector<Value> default_input_values_{3, 2, 1, 0};
};

TEST_F(
    NeuralNetworkBaseTestFixture,
    GivenInputValues_WhenSettingNumberOfNeuronsInInputLayer_ThenCorrectSizeIsSet) {

  const std::size_t expected_size{4};

  SetNumberOfNeuronsInInputLayer(default_input_values_.size());

  ASSERT_EQ(expected_size, input_layer_.capacity());
}

TEST_F(
    NeuralNetworkBaseTestFixture,
    GivenEmptyInputValues_WhenSettingNumberOfNeuronsInInputLayer_ThenExceptionIsThrown) {

  const std::size_t invalid_size{0};

  ASSERT_THROW(SetNumberOfNeuronsInInputLayer(invalid_size),
               std::invalid_argument);
}

//TEST_F(NeuralNetworkBaseTestFixture,
//       GivenTwoNeurons_WhenCreating_ThenNeuronsAreLinked) {
//
//  std::unique_ptr<Neuron> parent_neuron{new Neuron{2}};
//  std::unique_ptr<Neuron> child_neuron{new Neuron{1}};
//
//  Synapse synapse{parent_neuron.get(), child_neuron.get(), 0};
//
//  EXPECT_EQ(true, *synapse.parent);
//  EXPECT_EQ(true, *synapse.child);
//}

} // namespace Base

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
