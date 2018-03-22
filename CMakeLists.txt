cmake_minimum_required(VERSION 3.0)
project(mpi_sum)

find_package(Boost COMPONENTS mpi serialization REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(OPTS -Wall -Wextra -Werror -march=native -O3)

add_executable(mpi_sum main.cpp)
target_compile_options(mpi_sum PUBLIC ${OPTS})
target_link_libraries(mpi_sum PUBLIC PUBLIC ${Boost_LIBRARIES})
target_include_directories(mpi_sum PUBLIC ${Boost_INCLUDE_DIRS})