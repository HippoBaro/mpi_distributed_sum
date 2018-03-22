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
        if (comm.rank() > 0) { comm.send(root, 0, in_values, n); }
        else {
            auto response = std::vector<int, boost::simd::allocator<int>>(n);
            std::copy_n(in_values, n, out_values);
            for (int j = 0; j < comm.size() - 1; ++j) {
                comm.recv(boost::mpi::any_source, boost::mpi::any_tag, response.data(), n);
                std::transform(response.data(), response.data() + n, out_values, out_values, op);
            }
        }
    }
};

struct smarter_reduce {
    static constexpr auto name = "smarter_reduce";
    template<typename T, typename Op>
    void operator()(const boost::mpi::communicator &comm, T *in_values, int n, T *out_values, Op op, int root) {
        auto response = std::vector<int, boost::simd::allocator<int>>(n);

        auto recv_count = comm.rank() == comm.size() / 2 ? log2(comm.rank()) : log2(abs(comm.rank() - comm.size() / 2));
        int j = 0;
        for (; comm.rank() % 2 && j < recv_count; ++j) {
            comm.recv(comm.rank() + (j == 0 ? 1 : j + j), boost::mpi::any_tag, response.data(), n);
            std::transform(response.data(), response.data() + n, in_values, in_values, op);
        }
        if (comm.rank() != root) {
            comm.send(comm.rank() - (j == 0 ? 1 : j + j), 0, in_values, n);
        }
        else {
            std::copy_n(in_values, n, out_values);
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
auto reduce_and_accumulate(boost::mpi::communicator const &comm, Reducer reducer, Accumulator accumulator) {
    auto local = std::vector<int, boost::simd::allocator<int>>(Size);
    auto reduced = std::vector<int, boost::simd::allocator<int>>(Size);

    srand(1);  // useful for testing
    std::generate(local.begin(), local.end(), [] { return rand(); });

    comm.barrier();
    auto reduce_time = time_function([&] {
        reducer(comm, &local.front(), static_cast<int>(local.size()), &reduced.front(), std::plus<>(), 0);
    });
    if (comm.rank() > 0) { return std::make_pair(0, 0); }
    auto accumulate_time = time_function([&] {
        //volatile __attribute__((unused)) auto t = accumulator(&reduced.front(), &reduced.back(), 0);
        auto t = accumulator(&reduced.front(), &reduced.back(), 0);
        if (comm.rank() == 0)
            std::cout << t << std::endl;
    });

    return std::make_pair((int)reduce_time.count(), (int)accumulate_time.count());
}

template<size_t RoundCount, typename Function>
auto make_stat(Function &&function) {
    auto res = std::array<decltype(function()), RoundCount>();
    std::generate(res.begin(), res.end(), [&function] { return function(); });
    int min_reduce = std::numeric_limits<int>::max();
    int min_accumulate = std::numeric_limits<int>::max();

    for (auto &&item : res) {
        if (item.first < min_reduce) { min_reduce = item.first; }
        if (item.second < min_accumulate) { min_accumulate = item.second; }
    }
    return std::make_pair(min_reduce, min_accumulate);
}

template<size_t Size, typename Reducer, typename Accumulator>
void benchmark(boost::mpi::communicator const &comm) {
    make_stat<50>([&comm] { return reduce_and_accumulate<Size>(comm, Reducer(), Accumulator()); });
    //if (comm.rank() > 0) { return; }
    //std::cout << "[Size: " << Size << "] " << Reducer::name << ": " << min.first << "us, "
    //          << Accumulator::name << ": " << min.second << "us" << std::endl;
    std::cout << std::endl;
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

    benchmark<4194304, MPI_reduce, SIMD_accumulator>(world);
    benchmark<4194304, dumb_reduce, SIMD_accumulator>(world);
    benchmark<4194304, smarter_reduce, SIMD_accumulator>(world);
    return 0;
}