cmake_minimum_required(VERSION 3.0)
project(mpi_sum)

set(Boost_USE_STATIC_LIBS ON)
find_package(Boost COMPONENTS mpi serialization REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(OPTS -march=native -Ofast -qopenmp -parallel -Wall -Wextra -Werror -std=c++14)

add_executable(mpi_sum main.cpp)
target_compile_options(mpi_sum PUBLIC ${OPTS})
target_link_libraries(mpi_sum PUBLIC ${Boost_LIBRARIES} qopenmp)
target_include_directories(mpi_sum PUBLIC ${Boost_INCLUDE_DIRS} boost.simd/include)