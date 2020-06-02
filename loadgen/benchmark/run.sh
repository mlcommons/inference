#! /usr/bin/bash
echo "Building loadgen..."
if [ ! -e loadgen_build ]; then mkdir loadgen_build; fi;
cd loadgen_build && cmake ../.. && make -j && cd ..
echo "Building test program..."
if [ ! -e build ]; then mkdir build; fi;
g++ --std=c++11 -O3 -I.. -o build/repro.exe repro.cpp -Lloadgen_build -lmlperf_loadgen -lpthread && \
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so.2 build/repro.exe $1 $2 $3 $4