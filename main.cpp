#include <iostream>
#include <chrono>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include <boost/mpi.hpp>
#include <boost/simd.hpp>
#include <boost/simd/algorithm.hpp>
#include <boost/simd/memory.hpp>

#pragma GCC diagnostic pop

template<typename Function>
inline std::chrono::microseconds time_function(Function &&func) {
    auto begin_time = std::chrono::high_resolution_clock::now();
    func();
    return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - begin_time);
}

template<size_t Size>
void reduce_and_accumulate(boost::mpi::communicator const &world) {
    auto local_array = std::vector<int, boost::simd::allocator<int>>(Size);
    auto reduced_array = std::vector<int, boost::simd::allocator<int>>(Size);
    auto reduced_array_MPI = std::vector<int, boost::simd::allocator<int>>(Size);

    // the element of array is generated randomly
    std::generate(local_array.begin(), local_array.end(), [] { return rand(); });

    // MPI_Reduce and then print the usetime, the result will be put in res2[]
    {
        world.barrier();
        auto use_time2 = time_function([&] {
            boost::mpi::reduce(world, &local_array.front(), static_cast<int>(local_array.size()),
                               &reduced_array_MPI.front(), std::plus<>(), 0);
        });
        if (world.rank() == 0) {
            std::cout << Size << " int, use_time: " << use_time2.count() << "us [MPI_Reduce]\n";
        }
    }

    // YOUR_Reduce and then print the usetime, the result should be put in res[]
    {
        world.barrier();
        auto use_time = time_function([&] {
            boost::mpi::reduce(world, &local_array.front(), static_cast<int>(local_array.size()),
                               &reduced_array.front(), std::plus<>(), 0);
        });
        if (world.rank() == 0) {
            std::cout << Size << " int, use_time: " << use_time.count() << "us [YOUR_Reduce]\n";
        }
    }

    // check the result of MPI_Reduce and YOUR_Reduce
    if (world.rank() == 0) {
        auto correctness = boost::simd::equal(&reduced_array.front(), &reduced_array.back(), &reduced_array_MPI.front());
        if (!correctness) {
            printf("WRONG !!!\n");
        }
        else {
            printf("CORRECT !\n");
        }

        {
            uint64_t sum = 0;
            auto use_time = time_function([&] {
                sum = static_cast<uint64_t>(boost::simd::reduce(&reduced_array.front(), &reduced_array.back(), 0));
            });
            std::cout << "sum: " << sum << ", use_time: " << use_time.count() << "us [single thread]\n";
        }

        {
            uint64_t sum = 0;
            auto use_time = time_function([&] {
                sum = static_cast<uint64_t>(boost::simd::reduce(&reduced_array.front(), &reduced_array.back(), 0));
            });
            std::cout << "sum: " << sum << ", use_time: " << use_time.count() << "us [multiple threads]\n";
        }

        std::cout << std::endl;
    }
}

int main(int argc, char *argv[]) {
    boost::mpi::environment env{argc, argv};
    boost::mpi::communicator world;

    srand(1);  // useful for testing

    reduce_and_accumulate<4>(world);
    reduce_and_accumulate<64>(world);
    reduce_and_accumulate<1024>(world);
    reduce_and_accumulate<16384>(world);
    reduce_and_accumulate<262144>(world);
    reduce_and_accumulate<4194304>(world);
    return 0;
}