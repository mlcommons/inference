# This is a convenient way to build+install gn and ninja
# in the subdirectories of this project.
# It is not necessary to bootstrap if you have depot_tools
# or gn and ninja already installed on your system.
bootstrap_gn_ninja: gn ninja

.PHONY: ninja
ninja:
	cd third_party/ninja && python configure.py --bootstrap

.PHONY: gn
gn: ninja
	# Generate gn's ninja build file
	python third_party/gn/build/gen.py
	# Build gn
	third_party/ninja/ninja -C third_party/gn/out
	# Copy gn to third_party/gn, where depot_tools expects it.
	cp third_party/gn/out/gn* third_party/gn/.

.PHONY: mlperf_loadgen
mlperf_loadgen: gn
	third_party/gn/gn gen out/MakefileGnProj
	third_party/ninja/ninja -C out/MakefileGnProj mlperf_loadgen

.PHONY: mlperf_loadgen_pymodule
mlperf_loadgen_pymodule: gn
	third_party/gn/gn gen out/MakefileGnProj
	CFLAGS="-std=c++14" third_party/ninja/ninja -C out/MakefileGnProj loadgen_pymodule_wheel_src
