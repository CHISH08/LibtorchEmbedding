cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

# ${PROJECT_NAME}
project(CBOW)
list(APPEND CMAKE_PREFIX_PATH "~/code/Embedding/libtorch")

find_package(Torch REQUIRED)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include_directories(include)

file(GLOB
        CPPS "src/*.cpp"
        C "src/*.c")

add_executable(cbow ${CPPS})
target_link_libraries(cbow "${TORCH_LIBRARIES}")
set_property(TARGET cbow PROPERTY CXX_STANDARD 17)
