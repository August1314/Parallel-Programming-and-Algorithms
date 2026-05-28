#ifndef LAB6_PARALLEL_FOR_HPP
#define LAB6_PARALLEL_FOR_HPP

using parallel_for_functor = void *(*)(int, void*);

enum ParallelForSchedule {
    PARALLEL_FOR_BLOCK = 0,
    PARALLEL_FOR_CYCLIC = 1,
    PARALLEL_FOR_DYNAMIC = 2,
};

extern "C" void parallel_for(int start, int end, int inc, parallel_for_functor functor, void* arg, int num_threads);

extern "C" void parallel_for_with_schedule(
    int start,
    int end,
    int inc,
    parallel_for_functor functor,
    void* arg,
    int num_threads,
    ParallelForSchedule schedule,
    int chunk_size
);

ParallelForSchedule parse_parallel_for_schedule(const char* raw);
const char* parallel_for_schedule_name(ParallelForSchedule schedule);

#endif
