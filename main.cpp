#include <iostream>
#include <sys/time.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <boost/mpi.hpp>
#include <boost/simd.hpp>
#include <boost/simd/algorithm.hpp>
#include <boost/simd/memory.hpp>
#pragma GCC diagnostic pop

#define MAX_LEN 4194304

long get_time_us()  // return the time in the unit of us
{
    struct timeval my_time;  //us
    gettimeofday(&my_time, NULL);
    long runtime_us = 1000000 * my_time.tv_sec + my_time.tv_usec; // us
    return runtime_us;
}

int main(int argc, char *argv[])
{
    boost::mpi::environment env{argc, argv};
    boost::mpi::communicator world;

    srand(time(NULL));  // seed for rand() to generate the array randomly

    int count;
    long begin_time, end_time, use_time, use_time2; // use_time for YOUR_Reduce & use_time2 for MPI_Reduce

    int i;

    // initialize
    auto a = std::vector<int, boost::simd::allocator<int>>(MAX_LEN); // the array used to do the reduction
    auto res = std::vector<int, boost::simd::allocator<int>>(MAX_LEN); // the array to record the result of YOUR_Reduce
    auto res2 = std::vector<int, boost::simd::allocator<int>>(MAX_LEN); // the array to record the result of MPI_Reduce
    std::fill(a.begin(), a.end(), 0);
    std::fill(res.begin(), res.end(), 0);
    std::fill(res2.begin(), res2.end(), 0);

    // TODO
    // you can add some variable or some other things as you want if needed
    // TODO

    for(count=4; count<=MAX_LEN; count*=16) // length of array : [ 4  64  1024  16384  262144  4194304 ]
    {
        // the element of array is generated randomly
        for(i=0; i<MAX_LEN; i++)
        {
            a[i] = rand() % MAX_LEN;
        }

        // MPI_Reduce and then print the usetime, the result will be put in res2[]
        world.barrier();
        begin_time = get_time_us();
        boost::mpi::reduce(world, &a.front(), a.size(), &res2.front(), std::plus<>(), 0);
        world.barrier();
        end_time = get_time_us();
        use_time2 = end_time - begin_time;
        if(world.rank() == 0)
            printf("%d int use_time : %ld us [MPI_Reduce]\n", count, use_time2);


        // YOUR_Reduce and then print the usetime, the result should be put in res[]
        world.barrier();
        begin_time = get_time_us();
        // TODO
        // you should delete the next line, and do the reduction using your idea
        boost::mpi::reduce(world, &a.front(), a.size(), &res.front(), std::plus<>(), 0);
        // TODO
        world.barrier();
        end_time = get_time_us();
        use_time = end_time - begin_time;
        if(world.rank() == 0)
            printf("%d int use_time : %ld us [YOUR_Reduce]\n", count, use_time);


        // check the result of MPI_Reduce and YOUR_Reduce
        if(world.rank() == 0)
        {
            auto correctness = boost::simd::equal(res.data(), res.data() + res.size(), res2.data());
            if(!correctness)
                printf("WRONG !!!\n");
            else
                printf("CORRECT !\n");


            long sum = 0;
            begin_time = get_time_us();
            sum = boost::simd::reduce(res.data(), res.data() + res.size(), 0);
            end_time = get_time_us();
            use_time = end_time - begin_time;
            printf("sum is %ld, use_time : %ld us [single thread]\n", sum, use_time);

            sum = 0;
            begin_time = get_time_us();
            sum = boost::simd::reduce(res.data(), res.data() + res.size(), 0);
            end_time = get_time_us();
            use_time = end_time - begin_time;
            printf("sum is %ld, use_time : %ld us [multiple threads]\n\n", sum, use_time);
        }
    }
    return 0;
}