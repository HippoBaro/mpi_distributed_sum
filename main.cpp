#include <iostream>
#include <chrono>
#include <numeric>
#include <valarray>
#include <iomanip>

// On GCC 4.9, boost::mpi has warnings and -Werror prevents compilation
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <boost/mpi.hpp>
#pragma GCC diagnostic pop

#define likely(x) __builtin_expect ((x), 1)
#define unlikely(x) __builtin_expect ((x), 0)

/*
 * MPI sum.
 * Reduce an array scattered throughout multiple machines using MPI_Reduce and
 * then compute the sum of the reduced array
 */

/// Simple SIMDed accumulator, reduce an array to a single value using std::plus<>
struct SIMD_accumulator {
    static constexpr auto name = "SIMD_accumulator";

    template<typename T, typename BinaryOperation>
    __attribute__((always_inline)) auto operator()(T const * __restrict__ first, T const * __restrict__ last, T init,
                                                   BinaryOperation op) {
        return std::accumulate(first, last, init, op);
    }
};

/// Simple, non-optimized linear accumulator, reduce an array to a single value using std::plus<>
struct dumb_accumulator {
    static constexpr auto name = "dumb_accumulator";

    template<typename T, typename BinaryOperation>
    auto operator()(T const *first, T const *last, T init, BinaryOperation op) {
        volatile T res = init;
        while (first != last) { res = op(res, *first++); }
        return res;
    }
};

/// OpenMP accumulator. OpenMP only creates thread when required so this isn't much faster that the SIMD version
struct parallel_accumulator {
    static constexpr auto name = "parallel_accumulator";

    template<typename T, typename BinaryOperation>
    auto operator()(T const * __restrict__ first, T const * __restrict__ last, T init, BinaryOperation op) {
        T res = init;
        #pragma omp parallel for reduction (+:res) schedule(static)
        for (int j = 0; j < std::distance(first, last); ++j) {
            res = op(res, first[j]);
        }
        return res;
    }
};

/// Reducer that uses the native MPI_Reduce function
/// \tparam Size Size of the arrays to reduce
template <size_t Size>
struct MPI_reduce {
    static constexpr auto name = "MPI_reduce";

    template<typename T, typename Op>
    void operator()(const boost::mpi::communicator &comm, const T *in_values, T *out_values, Op op, int root) {
        boost::mpi::reduce(comm, in_values, Size, out_values, op, root);
    }
};

std::vector<int> responses(4194304);

/// Very simple reducer where every node except root sends it's part of the information to root.
/// Root reduces as it receives them
/// \tparam Size Size of the arrays to reduce
template <size_t Size>
struct dumb_reduce {
    static constexpr auto name = "dumb_reduce";

    template<typename T, typename Op>
    void operator()(const boost::mpi::communicator &comm, T *__restrict__ in_values, T * __restrict__ out_values, Op op, int root) {
        if (comm.rank() == root) {
            std::memcpy(out_values, in_values, Size * sizeof(T));
            for (int j = 0; j < comm.size() - 1; ++j) {
                comm.recv(boost::mpi::any_source, boost::mpi::any_tag, responses.data(), Size);
                std::transform(out_values, out_values + Size, responses.data(), out_values, op);
            }
        }
        else { comm.send(root, 0, in_values, Size); }
    }
};

///We need the log2 function to make binomial tree, but log2() is freakin slow
constexpr std::array<int, 9> log2_a{ {-1, 0, 1, -1, 2, -1, -1, -1, 3} };

/// Smarter reducer implementing a binomial-tree
/// A  AB   ABCD
/// B /    /
/// C  CD /
/// D /
/// \tparam Size Size of the arrays to reduce
template<size_t Size>
struct smarter_reduce : public dumb_reduce<Size> {
    static constexpr auto name = "smarter_reduce";

    template<typename T, typename Op>
    __attribute__((always_inline)) void impl(const boost::mpi::communicator &comm, T * __restrict__ in_values,
                                             T * __restrict__ out_values, Op op, int root)  {
        if (likely (!(comm.rank() % 2))) {
            comm.recv(comm.rank() + 1, boost::mpi::any_tag, responses.data(), Size);
            std::transform(in_values, in_values + Size, responses.data(), out_values, op);
        }

        int recv_count;
        if (unlikely(comm.rank() == root)) { recv_count = log2_a[comm.size()]; }
        else { recv_count = comm.rank() == comm.size() / 2 ? log2_a[comm.rank()] : log2_a[abs(comm.rank() - comm.size() / 2)]; }

        int j = 0;
        if (likely (recv_count > 0)) {
            for (j=1; !(comm.rank() % 2) && j < recv_count; ++j) {
                comm.recv(comm.rank() + (j == 0 ? 1 : j + j), boost::mpi::any_tag, responses.data(), Size);
                std::transform(out_values, out_values + Size, responses.data(), out_values, op);
            }
        }
        if (likely (comm.rank() != root)) {
            MPI_Send((recv_count > 0) ? out_values : in_values, Size, boost::mpi::get_mpi_datatype<T>(*out_values),
                     comm.rank() - (j == 0 ? 1 : j + j), 0, comm);
        }
    }

    template<typename T, typename Op>
    __attribute__((always_inline)) void operator()(const boost::mpi::communicator &comm,T * __restrict__ in_values,
                                                   T * __restrict__ out_values, Op op, int root)  {
        /// If the arrays are smaller than the standard MTU, there is no practical advantages paying the overhead
        /// of reducing via a binomial-tree, so we fallback to the dumb_reducer to improve latency.
        if (Size > 1500) { impl(comm, in_values, out_values, op, root); }
        else { dumb_reduce<Size>::operator()(comm, in_values, out_values, op, root); }
    }
};

/// Utility function to time the execution of a function using the most precise clock available on the system
/// \tparam Function The function's type
/// \param func the callable object
/// \return a std::chrono duration casted as microsoconds.
template<typename Function>
inline std::chrono::microseconds time_function(Function &&func) {
    auto begin_time = std::chrono::high_resolution_clock::now();
    func();
    return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - begin_time);
}

/// Main benchmark function. Reduce and accumulate distributed vectors.
/// \tparam Size Size of the arrays to work with
/// \tparam Reducer Reducing policy
/// \tparam Accumulator Accumulating policy
/// \param comm Communicator to benchmark on
/// \param reducer instance of the reducer
/// \param accumulator instance of the accumulator
/// \return a pair containg the execution time for the reduction and accumulation
template<size_t Size, typename Reducer, typename Accumulator>
auto reduce_and_accumulate(boost::mpi::communicator const &comm, Reducer reducer, Accumulator accumulator) {
    static std::vector<int> local(4194304);
    static std::vector<int> reduced(4194304);

    srand(1);  // useful for testing the results of the final accumulations accros Reducers and Accumulators
    std::generate(local.begin(), local.begin() + Size, [] { return rand(); });

    comm.barrier(); // This barraiers ensures that all nodes are synchronized.
    auto reduce_time = time_function([&] { reducer(comm, &local.front(), &reduced.front(), std::plus<>(), 0); });
    if (likely (comm.rank() > 0)) { return std::make_tuple(0, 0, 0); }
    int accumulation_result;
    auto accumulate_time = time_function([&] {
        accumulation_result = accumulator(reduced.data(), reduced.data() + Size, 0, std::plus<>());
    });

    return std::make_tuple((int)reduce_time.count(), (int)accumulate_time.count(), accumulation_result);
}

struct result {
    struct minmaxavg {
        int min;
        int max;
        float avg;
    };
    struct minmaxavg reduction;
    struct minmaxavg accumulation;
    int accumulation_sum;
};

/// Execute benchmark function multiple times and select the best result
/// \tparam RoundCount How many time to execute the function
/// \tparam Function Benchmark function type
/// \param function Benchmark function
/// \return a pair containg the best result for reducing and accumulating
template<size_t RoundCount, typename Function>
auto make_stat(Function &&function) {
    static std::array<decltype(function()), RoundCount> res;
    std::generate(res.begin(), res.end(), [&function] { return function(); });
    struct result result{{ std::numeric_limits<int>::max(), std::numeric_limits<int>::min(), 0 },
                         { std::numeric_limits<int>::max(), std::numeric_limits<int>::min(), 0 },
                         std::get<2>(res.front()) };

    int sum_reduction = 0, sum_accumulation = 0;
    for (auto &&item : res) {
        if (std::get<0>(item) < result.reduction.min) { result.reduction.min = std::get<0>(item); }
        if (std::get<1>(item) < result.accumulation.min) { result.accumulation.min = std::get<1>(item); }
        if (std::get<0>(item) > result.reduction.max) { result.reduction.max = std::get<0>(item); }
        if (std::get<1>(item) > result.accumulation.max) { result.accumulation.max = std::get<1>(item); }
        sum_reduction += std::get<0>(item);
        sum_accumulation += std::get<1>(item);
    }
    result.reduction.avg = sum_reduction / res.size();
    result.accumulation.avg = sum_accumulation / res.size();
    return result;
}

/// Retreive the results and format result
/// \tparam Size Size of the array to work with
/// \tparam Reducer Reducing policy
/// \tparam Accumulator Accumulating policy
/// \param comm Communicator to benchmark on
template<size_t Size, template<size_t> class Reducer, typename Accumulator>
int benchmark(boost::mpi::communicator const &comm) {
    struct result results = make_stat<100>([&comm] { return reduce_and_accumulate<Size>(comm, Reducer<Size>(), Accumulator()); });
    if (comm.rank() > 0) { return 0; }
    std::cout << std::setw(10) << std::left  << Size << std::setw(10) << std::right << results.reduction.min
              << std::setw(10) << std::right << results.reduction.max << std::setw(10) << std::right << results.reduction.avg
              << std::setw(10) << std::right << results.accumulation.min << std::setw(10) << std::right << results.accumulation.max
              << std::setw(10) << std::right << results.accumulation.avg << std::setw(14) << std::right << results.accumulation_sum << std::endl;
    return 0;
}

/// Expand/Unroll the variadic Size template parametter and launch benchmarks for each of the specified array size
/// \tparam Reducer Reducing policy
/// \tparam Accumulator Accumulating policy
/// \tparam Size Sizes of the arrays to work with
/// \param comm Communicator to benchmark on
template<template<size_t> class Reducer, typename Accumulator, size_t ...Size>
void unroll_benchmark(boost::mpi::communicator const &comm) {
    using expander = int[];
    if (comm.rank() == 0) {
        std::cout << "Testing using " << Reducer<0>::name << " and " << Accumulator::name << " [" << 100 << " rounds]:\n";
        std::cout << std::setw(10) << std::left  << "Data size" << std::setw(10) << std::right << "Red. min"
                  << std::setw(10) << std::right << "Red. max" << std::setw(10) << std::right << "Red. avr"
                  << std::setw(10) << std::right << "Acc. min" << std::setw(10) << std::right << "Acc. max"
                  << std::setw(10) << std::right << "Acc. avr" << std::setw(14) << std::right << "Result" << std::endl;
    }
    (void) expander{ 0, (benchmark<Size, Reducer, Accumulator>(comm))... };
    if (comm.rank() > 0) { return; }
    std::cout << std::endl;
}

int main(int argc, char *argv[]) {
    boost::mpi::environment env{argc, argv};
    boost::mpi::communicator world;

    unroll_benchmark<MPI_reduce, SIMD_accumulator, 4, 64, 1024, 16384, 262144, 4194304>(world);
    unroll_benchmark<dumb_reduce, SIMD_accumulator, 4, 64, 1024, 16384, 262144, 4194304>(world);
    unroll_benchmark<smarter_reduce, SIMD_accumulator, 4, 64, 1024, 16384, 262144, 4194304>(world);
    return EXIT_SUCCESS;
}