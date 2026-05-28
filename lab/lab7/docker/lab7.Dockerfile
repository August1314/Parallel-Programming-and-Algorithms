FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        openmpi-bin \
        libopenmpi-dev \
        valgrind \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/lab/lab7
