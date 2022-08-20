cmake_minimum_required(VERSION 3.5)

project(demo)
set(CMAKE_CXX_STANDARD 14)

add_definitions(-DHAND)
add_definitions(-g)

include_directories(
    ${PROJECT_SOURCE_DIR}/../include
)

file(GLOB_RECURSE LEAN_FILE ${PROJECT_SOURCE_DIR}/../src/*.cc)
message(${LEAN_FILE})


add_executable(demo
    ${LEAN_FILE}
    ./mnist.cc
)

target_link_libraries(demo
    libprotobuf.a
    pthread
)

