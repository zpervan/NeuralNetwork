project(Libraries)

add_library(NeuralNetworkBase
        STATIC
        neural_network_base.h
        neural_network_base.cpp
        )

add_library(FeedForwardNetwork
        STATIC
        feed_forward_network.h
        feed_forward_network.cpp
        )

target_link_libraries(NeuralNetworkBase
        FeedForwardNetwork
        )

###### TESTING ######
add_subdirectory(ThirdParty)

enable_testing()

set(GTest_Libs
        gtest
        gtest_main
        )

# Add test executables
add_executable(FeedForwardNetworkTest
        Test/feed_forward_network_test.cpp
        )

# Link libraries needed for testing

target_link_libraries(FeedForwardNetworkTest
        FeedForwardNetwork
        ${GTest_Libs}
        )

# Create a test suite
add_test(TestAll
        FeedForwardNetworkTest
        )
