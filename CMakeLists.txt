cmake_minimum_required(VERSION 3.10)

project(Embeddings)

add_subdirectory(CBOW)

add_executable(main main.cpp)

target_link_libraries(main CBOW)
