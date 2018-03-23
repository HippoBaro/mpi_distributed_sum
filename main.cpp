#include <iostream>
#include <chrono>
#include <numeric>
#include <valarray>

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
    auto operator()(T const * __restrict__ first, T const * __restrict__ last, T init) {
        return std::accumulate(first, last, init);
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
                for (volatile int k = 0; k < n; ++k) {
                    out_values[k] = op(response.data()[k], out_values[k]);
                }
            }
        }
    }
};

struct smarter_reduce {
    static constexpr auto name = "smarter_reduce";
    template<typename T, typename Op>
    void operator()(const boost::mpi::communicator &comm, T * __restrict__ in_values, int n, T * __restrict__ out_values, Op op, int root) {
        static std::vector<boost::mpi::request> pending(3);
        int recv_count;
        static std::vector<int, boost::simd::allocator<int>> responses(4194304 * 3);
        if (comm.rank() == root) { recv_count = (int)log2(comm.size()); }
        else { recv_count = comm.rank() == comm.size() / 2 ? (int)log2(comm.rank()) : (int)log2(abs(comm.rank() - comm.size() / 2)); }

        int j = 0;
        //if (recv_count > 0) {
        for (; !(comm.rank() % 2) && j < recv_count; ++j) {
            pending[j] = comm.irecv(comm.rank() + (j == 0 ? 1 : j + j), boost::mpi::any_tag, responses.data() + n * j, n);
        }
        memcpy(out_values, in_values, n * sizeof(int));
        for (int k = 0; k < recv_count; ++k) {
            auto p = boost::mpi::wait_any(pending.begin(), pending.end());
            auto index = std::distance(pending.begin(), p.second);
            std::transform(out_values, out_values + n, responses.data() + n * index, out_values, op);
        }
        //}
        if (comm.rank() != root) {
            if (n < 16) {
                MPI_Send(out_values, n, boost::mpi::get_mpi_datatype<T>(*out_values), comm.rank() - (j == 0 ? 1 : j + j), 0, comm);
            }
            else {
                MPI_Rsend(out_values, n, boost::mpi::get_mpi_datatype<T>(*out_values), comm.rank() - (j == 0 ? 1 : j + j), 0, comm);
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
        volatile __attribute__((unused)) auto t = accumulator(&reduced.front(), &reduced.back(), 0);
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
    auto min = make_stat<500>([&comm] { return reduce_and_accumulate<Size>(comm, Reducer(), Accumulator()); });
    if (comm.rank() > 0) { return; }
    std::cout << "[Size: " << Size << "] " << Reducer::name << ": " << min.first << "us, "
              << Accumulator::name << ": " << min.second << "us" << std::endl;
}

int main(int argc, char *argv[]) {
    boost::mpi::environment env{argc, argv};
    boost::mpi::communicator world;

    benchmark<4, dumb_reduce, SIMD_accumulator>(world);
    benchmark<4, smarter_reduce, SIMD_accumulator>(world);
    benchmark<4, MPI_reduce, SIMD_accumulator>(world);

    benchmark<64, dumb_reduce, SIMD_accumulator>(world);
    benchmark<64, smarter_reduce, SIMD_accumulator>(world);
    benchmark<64, MPI_reduce, SIMD_accumulator>(world);

    benchmark<1024, dumb_reduce, SIMD_accumulator>(world);
    benchmark<1024, smarter_reduce, SIMD_accumulator>(world);
    benchmark<1024, MPI_reduce, SIMD_accumulator>(world);

    benchmark<16384, dumb_reduce, SIMD_accumulator>(world);
    benchmark<16384, smarter_reduce, SIMD_accumulator>(world);
    benchmark<16384, MPI_reduce, SIMD_accumulator>(world);

    benchmark<262144, dumb_reduce, SIMD_accumulator>(world);
    benchmark<262144, smarter_reduce, SIMD_accumulator>(world);
    benchmark<262144, MPI_reduce, SIMD_accumulator>(world);

    benchmark<4194304, dumb_reduce, SIMD_accumulator>(world);
    benchmark<4194304, smarter_reduce, SIMD_accumulator>(world);
    benchmark<4194304, MPI_reduce, SIMD_accumulator>(world);
    return 0;
}