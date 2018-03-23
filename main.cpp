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

#define likely(x) __builtin_expect ((x), 1)
#define unlikely(x) __builtin_expect ((x), 0)

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

std::vector<int, boost::simd::allocator<int>> responses(4194304);

struct dumb_reduce {
    static constexpr auto name = "dumb_reduce";

    template<typename T, typename Op>
    void operator()(const boost::mpi::communicator &comm, T *__restrict__ in_values, int n, T * __restrict__ out_values, Op op, int root) {
        if (unlikely(comm.rank() == root)) {
            for (int j = 0; j < comm.size() - 1; ++j) {
                comm.recv(boost::mpi::any_source, boost::mpi::any_tag, responses.data(), n);
                std::transform(in_values, in_values + n, responses.data(), in_values, op);
                for (int k = 0; k < n; ++k) {
                    out_values[k] = op(responses.data()[k], out_values[k]);
                }
            }
            std::memcpy(out_values, in_values, n);
        }
        else { comm.send(root, 0, in_values, n); }
    }
};

constexpr std::array<int, 9> log2_a{ {-1, 0, 1, -1, 2, -1, -1, -1, 3} };

struct smarter_reduce {
    static constexpr auto name = "smarter_reduce";
    template<typename T, typename Op>
    void operator()(const boost::mpi::communicator &comm, T * __restrict__ in_values, int n, T * __restrict__ out_values, Op op, int root) {
        if (likely (!(comm.rank() % 2))) {
            comm.recv(comm.rank() + 1, boost::mpi::any_tag, responses.data(), n);
            std::transform(in_values, in_values + n, responses.data(), out_values, op);
        }

        int recv_count;
        if (unlikely(comm.rank() == root)) { recv_count = log2_a[comm.size()]; }
        else { recv_count = comm.rank() == comm.size() / 2 ? log2_a[comm.rank()] : log2_a[abs(comm.rank() - comm.size() / 2)]; }

        int j = 0;
        if (likely (recv_count > 0)) {
            for (j=1; !(comm.rank() % 2) && j < recv_count; ++j) {
                comm.recv(comm.rank() + (j == 0 ? 1 : j + j), boost::mpi::any_tag, responses.data(), n);
                std::transform(out_values, out_values + n, responses.data(), out_values, op);
            }
        }
        if (likely (comm.rank() != root)) {
            MPI_Send((recv_count > 0) ? out_values : in_values, n, boost::mpi::get_mpi_datatype<T>(*out_values), comm.rank() - (j == 0 ? 1 : j + j), 0, comm);
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

std::vector<int, boost::simd::allocator<int>> local(4194304);
std::vector<int, boost::simd::allocator<int>> reduced(4194304);

template<size_t Size, typename Reducer, typename Accumulator>
auto reduce_and_accumulate(boost::mpi::communicator const &comm, Reducer reducer, Accumulator accumulator) {
    srand(1);  // useful for testing
    std::generate(local.begin(), local.begin() + Size, [] { return rand(); });

    comm.barrier();
    auto reduce_time = time_function([&] { reducer(comm, &local.front(), Size, &reduced.front(), std::plus<>(), 0); });
    if (likely (comm.rank() > 0)) { return std::make_pair(0, 0); }
    auto accumulate_time = time_function([&] {
        volatile __attribute__((unused)) auto t = accumulator(reduced.data(), reduced.data() + Size, 0);
    });

    return std::make_pair((int)reduce_time.count(), (int)accumulate_time.count());
}

template<size_t RoundCount, typename Function>
auto make_stat(Function &&function) {
    static std::array<decltype(function()), RoundCount> res;
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
    auto min = make_stat<100>([&comm] { return reduce_and_accumulate<Size>(comm, Reducer(), Accumulator()); });
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