# Building the MLPerf Inference Test Harness

The test harness library is currently built using the same set of tools that the
Chromium and related projects use:

* Dependencies are managed with [depot\_tools](https://commondatastorage.googleapis.com/chrome-infra-docs/flat/depot_tools/docs/html/depot_tools_tutorial.html#_setting_up).
* Project files are provided for the [gn metabuild tool](https://gn.googlesource.com/gn/+/master).
* Building uses [ninja](https://ninja-build.org/).
* Platform-specific toolchains used by ninja to create targets are pulled from Chromium's [build](https://chromium.googlesource.com/chromium/src/build) and [buildtools](https://chromium.googlesource.com/chromium/src/buildtools).
  * TODO: Figure out how to support many platforms without depending directly on Chromium like this.

If the tools above don't cover your particular configuration, please reach out. Patches to support other build environments welcome.

## Downloading the source

Download and install depot\_tools:

    git clone 'https://chromium.googlesource.com/chromium/tools/depot_tools.git'
    export PATH="${PWD}/depot_tools:${PATH}"

Copy the depot\_tools fetch config for the MLPerf test harness to the
depot\_tools path:

    wget https://raw.githubusercontent.com/mlperf/inference/master/depot_tools/fetch_configs/mlperf_inference.py -O ${PWD}/depot_tools/fetch_configs/mlperf_inference.py

Create a folder for the test harness project and fetch the source code:

    mkdir mlpi_test_harness
    cd mlpi_test_harness
    fetch mlperf_inference
    gclient sync

## Building from source

<i>Note: This has only been tested to work in a Debian-based linux environment so
far, but it should support many others.</i>

Run the gn metabuild. The output directory will contain ninja build files for a
specific target platform and set of build options.

    gn gen --root=src out/Default

TODO: Provide gn commands for cross-compiling to Android and to iOS. In the mean
time, looking at how to build chromium
[for android](https://chromium.googlesource.com/chromium/src/+/master/docs/android_build_instructions.md)
or [for ios](https://chromium.googlesource.com/chromium/src/+/HEAD/docs/ios/build_instructions.md)
should provide some useful pointers.

Optionally, create a project file for your favorite IDE. See [gn documentation](https://gn.googlesource.com/gn/+/master/docs/reference.md#ide-options) for details.

    gn gen --root=src --ide=eclipse out/Default
    gn gen --root=src --ide=vs out/Default
    gn gen --root=src --ide=xcode out/Default
    gn gen --root=src --ide=qtcreator out/Default

Build the library:

    ninja -C out/Default mlpi_test_harness

You will find the binary in out/Default/obj/test\_harness/libmlpi\_test\_harness.a. (Or as a .lib on Windows).

Link that library directly into the executable you want to test.

## Building as a Python module

TODO: It's not supported yet.

## Notes about important dependency and build files

**DEPS**: Describes source repositories to pull in as dependencies for the "gclient
sync" command. Also runs system commands to download the relevant toolchains
for the "gclient runhooks" command, which is run as part of the initial "fetch".

**.gn**: The root gn build file.

**BUILD.gn**: Located in multiple directories. Each one describes how to build
that particular directory.

## Build skia for image decode ???

Download and build skia separately as a static library and copy the resulting
binary to src/third\_party/skia/libs.

See [https://skia.org/user/build](https://skia.org/user/build]) for details.

TODO: Set up depot\_tools and gn to download and build skia as part of the test
harness in one go.

