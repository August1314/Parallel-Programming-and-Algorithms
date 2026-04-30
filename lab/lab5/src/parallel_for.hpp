#ifndef LAB5_PARALLEL_FOR_HPP
#define LAB5_PARALLEL_FOR_HPP

using parallel_for_functor = void *(*)(int, void*);

extern "C" void parallel_for(int start, int end, int inc, parallel_for_functor functor, void* arg, int num_threads);

#endif
