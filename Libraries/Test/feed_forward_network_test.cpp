#include "../feed_forward_network.h"

#include "gtest/gtest.h"

TEST(SimpleNeuralNetworkTest,
     GivenSimpleNeuralNetworkClass_WhenInstatiating_ThenObjectIsCreated) {
  auto *feed_forward_network = new FeedForwardNetwork();

  ASSERT_TRUE(feed_forward_network);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
