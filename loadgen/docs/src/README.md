# Generating the HTML docs

*Prerequisite:* You must have [doxygen](http://www.doxygen.nl) installed
on your system:

## With gn / ninja

If you are using the gn build flow, you may run:

    ninja -C out/Release generate_doxygen_html

* This will output the documentation to out/Release/gen/loadgen/docs/gen and
avoid poluting the source directory.

## Manually

Alternatively, you can manually run:

    python docs/src/doxygen_html_generator.py <target_dir> <loadgen_root>

* If <loadgen_root> is omittted, it will default to ".".
* If <target_dir> is also omitted, it will default to "./docs/gen".



