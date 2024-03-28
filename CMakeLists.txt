cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(EMBEDDING)

list(APPEND CMAKE_PREFIX_PATH "~/code/Embedding/libtorch")

find_package(Torch REQUIRED)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/CBOW/include
    ${CMAKE_CURRENT_SOURCE_DIR}/Embedding/include
)

file(GLOB SOURCES
    "CBOW/src/*.cpp"
    "Embedding/src/*.cpp"
    "main.cpp"
)

add_executable(embedding ${SOURCES})
target_link_libraries(embedding "${TORCH_LIBRARIES}")
set_property(TARGET embedding PROPERTY CXX_STANDARD 17)
