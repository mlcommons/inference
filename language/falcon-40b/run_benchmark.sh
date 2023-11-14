
#export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
#export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

python -u main.py --scenario Offline \
		--mlperf-conf mlperf.conf \
		--user-conf user.conf \
		--total-sample-count 1024 \
		--device cpu
