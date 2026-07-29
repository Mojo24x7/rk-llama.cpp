#!/bin/bash
# tp_shard is NOT a CMake target -- `make` will NEVER rebuild it.
# It MUST be rebuilt whenever tp_shard.cpp or the rktp wire protocol changes,
# otherwise the coordinator talks a newer protocol to an older shard: weights
# are never registered, MM replies ob=0, and compute() used to SIGSEGV.
# (That is exactly what masqueraded as the "rktp M>1 prefill crash".)
set -e
cd "$(dirname "$0")"
g++ -O2 -std=c++17 tp_shard.cpp -o build/bin/tp_shard \
    -I ggml/include -I ggml/src -L build/bin \
    -lggml -lggml-base -lpthread -Wl,-rpath,"$PWD/build/bin"
echo "tp_shard rebuilt: $(ls -l --time-style=+%F_%H:%M build/bin/tp_shard | awk "{print \$6}")"
echo "Now copy to every shard board:  rsync -a build/bin/tp_shard rock@<board>:/home/rock/rkllama-bin/"
