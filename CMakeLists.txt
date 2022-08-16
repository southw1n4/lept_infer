cmake_minimum_required(VERSION 3.5)

project(leptifner)
set(CMAKE_CXX_STANDARD 17)

add_definitions(-DHAND)
add_definitions(-g)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

file(GLOB_RECURSE LEAN_FILE ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB_RECURSE TEST_FILE ${PROJECT_SOURCE_DIR}/test/*.cc)
file(GLOB_RECURSE BENCH_FILE ${PROJECT_SOURCE_DIR}/benchmark/*.cc)


add_executable(infertest
    ${LEAN_FILE}
    ${TEST_FILE}
)

add_executable(inferbenchmark
    ${LEAN_FILE}
    ${BENCH_FILE}
)

target_link_libraries(infertest
    protobuf
)


add_custom_target(
    test
    DEPENDS infertest
    COMMAND ./infertest
)

add_custom_target(
    benchmark
    DEPENDS inferbenchmark
    COMMAND ./inferbenchmark
)
