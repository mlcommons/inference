# LoadGen FAQ {#ReadmeFAQ}

## Q: Can I make local modifications to the LoadGen for submission?
**A:** No. To keep the playing field level, please upstream any local
modificiations you need to make.

## Q: Where is version_generated.cc? I'm getting linker errors for LoadgenVersion definitions.
**A:** If you have a custom build setup, make sure you run the
version_generator.py script, which will create the cc file you are looking for.
The official build files that come with the LoadGen do this automatically for
you.

## Q: What is this version_generator.py script?
**A:** The LoadGen records git stats and the SHA1 of all its source files at
build time for verification purposes. This is easy to circumvent, but try your
best to run version_generator.py correctly; ideally integrated with your build
system if you have a custom build. The intention is more to help with debugging
efforts and detect accidental version missmatches than to detect bad actors.

## Q: How do I view the mlperf_log_trace.json file?
**A:** This file uses the [Trace Event Format]
(https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/edit)
to record a timeline of all the threads involved.
You can view the file by typing [chrome://tracing](chrome://tracing) into
Chrome's address bar and dragging the json file there.
This file zips well and you can drag the zip file directly into
"chrome:tracing" too.
Please include zipped traces (and the other logs) when filing bug reports.

## Q: What is the difference between the MultiStream and MultiStreamFree scenarios?
**A:** MultiStream corresponds to the official MLPerf scenario for submissions;
it has a fixed query rate and allows only one outstanding query at a time.
MultiStreamFree is implemented for evaluation purposes only; it sends queries
as fast as possible and allows up to N outstanding queries at a time. You may
want to use MultiStreamFree for development purposes since small improvements
in performance will always be reflected in the results, whereas MultiStream's
results will be quantized.
