#include <iostream>
#include <chrono>
#include <numeric>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include <boost/mpi.hpp>
#include <boost/simd.hpp>
#include <boost/simd/algorithm.hpp>
#include <boost/simd/memory.hpp>

#pragma GCC diagnostic pop

struct SIMD_accumulator {
    static constexpr auto name = "SIMD_accumulator";

    template<typename T>
    auto operator()(T const *first, T const *last, T init) {
        return boost::simd::reduce(first, last, init);
    }
};

struct dumb_accumulator {
    static constexpr auto name = "dumb_accumulator";

    template<typename T>
    auto operator()(T const *first, T const *last, T init) {
        volatile T res = init;
        while (first != last) { res += *first++; }
        return res;
    }
};

struct MPI_reduce {
    static constexpr auto name = "MPI_reduce";

    template<typename T, typename Op>
    void operator()(const boost::mpi::communicator &comm, const T *in_values, int n, T *out_values, Op op, int root) {
        boost::mpi::reduce(comm, in_values, n, out_values, op, root);
    }
};

struct dumb_reduce {
    static constexpr auto name = "dumb_reduce";

    template<typename T, typename Op>
    void operator()(const boost::mpi::communicator &comm, const T *in_values, int n, T *out_values, Op op, int root) {
        if (comm.rank() > 0) {
            comm.send(root, 0, in_values, n);
        }
        else { //receive and accumulate
            auto responses = std::vector<std::vector<int, boost::simd::allocator<int>>>(comm.size() - 1);
            std::generate(responses.begin(), responses.end(),
                          [n] { return std::vector<int, boost::simd::allocator<int>>(n); });

            std::copy_n(in_values, n, out_values);
            for (int j = 0; j < comm.size() - 1; ++j) {
                comm.recv(boost::mpi::any_source, boost::mpi::any_tag, responses[j].data(), n);
                std::transform(out_values, out_values + n, responses[j].data(), out_values, op);
            }
        }
    }
};

template<typename Function>
inline std::chrono::microseconds time_function(Function &&func) {
    auto begin_time = std::chrono::high_resolution_clock::now();
    func();
    return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - begin_time);
}

template<size_t Size, typename Reducer, typename Accumulator>
void reduce_and_accumulate(boost::mpi::communicator const &world, Reducer reducer, Accumulator accumulator) {
    auto local = std::vector<int, boost::simd::allocator<int>>(Size);
    auto reduced = std::vector<int, boost::simd::allocator<int>>(Size);

    srand(1);  // useful for testing
    std::generate(local.begin(), local.end(), [] { return rand(); });

    world.barrier();
    auto use_time = time_function([&] {
        reducer(world, &local.front(), static_cast<int>(local.size()), &reduced.front(), std::plus<>(), 0);
    });
    if (world.rank() > 0) { return; }

    std::cout << Size << " int, use_time: " << use_time.count() << "us [" << Reducer::name << "]\n";
    int sum = 0;
    auto use_time2 = time_function([&] { sum = accumulator(&reduced.front(), &reduced.back(), 0); });
    std::cout << "sum: " << sum << ", use_time: " << use_time2.count() << "us [" << Accumulator::name << "]\n"
              << std::endl;
}

int main(int argc, char *argv[]) {
    boost::mpi::environment env{argc, argv};
    boost::mpi::communicator world;

/*    reduce_and_accumulate<4>(world, MPI_reduce(), dumb_accumulator());
    reduce_and_accumulate<4>(world, MPI_reduce(), SIMD_accumulator());

    reduce_and_accumulate<64>(world, MPI_reduce(), dumb_accumulator());
    reduce_and_accumulate<1024>(world, MPI_reduce(), dumb_accumulator());
    reduce_and_accumulate<16384>(world, MPI_reduce(), dumb_accumulator());
    reduce_and_accumulate<262144>(world, MPI_reduce(), dumb_accumulator());*/

    reduce_and_accumulate<4194304>(world, MPI_reduce(), dumb_accumulator());

    reduce_and_accumulate<4194304>(world, MPI_reduce(), SIMD_accumulator());
    reduce_and_accumulate<4194304>(world, dumb_reduce(), SIMD_accumulator());
    return 0;
}