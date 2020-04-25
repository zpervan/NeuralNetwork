cmake_minimum_required(VERSION 3.12)

# Project name
project(NeuralNetwork CXX)

# Define C++ standard
set(CMAKE_CXX_STANDARD 17)

# Add needed subdirectories
add_subdirectory(Libraries)

add_executable(Runner runner.cpp)

target_link_libraries(Runner
        PRIVATE
        FeedForwardNetwork
        )
