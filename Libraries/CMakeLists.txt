project(Libraries)

add_subdirectory(Base)

# Header only libraries
add_library(ActivationFunctions
        STATIC
        activation_functions.h
        )

set_target_properties(ActivationFunctions
        PROPERTIES
        LINKER_LANGUAGE CXX
        )

# Libraries
add_library(NeuralNetworkBase
        STATIC
        neural_network_base.h
        neural_network_base.cpp
        )

target_link_libraries(NeuralNetworkBase
        PRIVATE
        ActivationFunctions
        ExclusiveOrNeuralNetwork
        NeuralNetworkComponents
        )

add_library(ExclusiveOrNeuralNetwork
        STATIC
        exclusive_or_neural_network.h
        exclusive_or_neural_network.cpp
        )

target_link_libraries(ExclusiveOrNeuralNetwork
        PRIVATE
        NeuralNetworkBase
        )

###### TESTING ######
add_subdirectory(ThirdParty)

enable_testing()

set(GTest_Libs
        gtest
        gtest_main
        )

# Add test executables
add_executable(ActivationFunctionsTest
        Test/activation_functions_test.cpp
        )

add_executable(ExclusiveOrNeuralNetworkTest
        Test/exclusive_or_neural_network_test.cpp
        )

add_executable(NeuralNetworkBaseTest
        Test/neural_network_base_test.cpp
        )

# Link libraries needed for testing
target_link_libraries(ActivationFunctionsTest
        ActivationFunctions
        ${GTest_Libs}
        )

target_link_libraries(ExclusiveOrNeuralNetworkTest
        ExclusiveOrNeuralNetwork
        ${GTest_Libs}
        )

target_link_libraries(NeuralNetworkBaseTest
        NeuralNetworkBase
        ${GTest_Libs}
        )

# Create a test suite
add_test(TestAll
        ActivationFunctionsTest
        ExclusiveOrNeuralNetworkTest
        NeuralNetworkBaseTest
        )
