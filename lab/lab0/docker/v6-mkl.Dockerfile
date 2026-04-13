FROM intel/oneapi-basekit:latest

WORKDIR /workspace

COPY src/matmul_common.hpp /workspace/src/matmul_common.hpp
COPY src/v6_mkl.cpp /workspace/src/v6_mkl.cpp

RUN bash -lc "source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 && \
    icpx -O3 -std=c++17 -qmkl /workspace/src/v6_mkl.cpp -o /workspace/v6_mkl"

ENTRYPOINT ["/workspace/v6_mkl"]
