# MLPerf Inference Synthetic Load Test (SyLT) Harness

This is a repository of a Trace Generation and Replay Harness. The harness
allows system implementors to determine the maximum sustainable QPS subject to
latency constraints for various MLPerf models on their inference systems.

A Trace is a list of timestamp-query pairs. Real traces can be captured from
production systems. Trace generators produce synthetic traces. This trace
generator selects queries from a query library uniformly at random with
replacement and selects timestamps based on a Poisson process. The trace
generator can produce synthetic traces according to various configurable
parameters including: average QPS, length, and wall clock duration. The QPS of a
trace is the average rate of query arrival, which may not reflect the achieved
throughput (also measured in QPS) for any real implementation.

A Load Test combines a Trace with a inference system to produce a latency
bound. The arrival of each query in the trace is simulated at the query's
corresponding timestamp.

## Theory of Operations

1. Generate a trace with Poisson distributed arrival times satisfying all
   specified requirements in terms of queries, QPS, and duration.

2. Conduct a load test experiment using the trace. Measure query latency. Query
   latency is the difference between query arrival and completion of a query's
   inference.

3. Increase QPS until the latency constraint is violated. Use binary search to
   find the maximum QPS achieved without violating the latency constraint.

## Project Status

This code is very rough. I am looking for a high-level review and feedback from
the broader MLPerf inference community before polishing. Patches very welcome!

## FAQ

1. For online cloud inference, why measure maximum sustained throughput subject
   to high-percentile latency bounds? Why not throughput-only? Why not
   latency-only?

   When launching an online cloud inference service, engineers ultimately care
   about TCO/QPS subject to a latency bound. Since TCO/machine is known,
   QPS/machine must be measured. The maximum QPS sustainable by a machine is
   determined by setting up a test server and increasing load until the machine
   exceeds an application specific high-percentile latency bound. At low QPS,
   latency is relatively flat and at too high QPS, the machine is rapidly
   overloaded and grows until the machine is forced to drop requests. In between
   these two extremes, latency can vary chaotically. Long running tests are
   necessary to ensure that a machine can serve its maximum rated QPS
   indefinitely.
   
   Throughput-only measurements ignore the latency constraints placed on real
   systems. Online cloud systems may run at sub-optimal batch sizes or run
   mostly empty batch inferences in order to meet latency
   requirements. Throughput-only measurements are probably the right metrics for
   Cloud and Edge Batch inference workloads, because they don't have latency
   constraints.

   Latency-only measurements don't answer questions about the cost of scaling
   QPS. From an online cloud inference perspective, reducing latency rapidly
   reaches a point of diminishing returns. Online cloud users can't tell the
   difference between 1 ms inferences and 1 us inferences. Latency-only
   measurements are probably the right metrics for some Online Edge inference
   workloads.

2. Why Poisson? Why not constant arrival rate?

   Poisson models discrete random processes where every event is independent.
   Radioactive decay is Poisson distributed.  Poisson distributions are
   relatively easy to generate. Most people believe that query arrivals are
   Poisson distributed in real life. Most results of queueing theory apply to
   Poisson distributions. Alternatives to Poisson are more complicated to
   calculate, have fewer nice mathematical properties, and don’t differ much
   from Poisson anyway. Empirically, testing inference systems with constant
   rate traces yields overly optimistic maximum QPS measurements.

3. What is start up time?

   Start up time is the time between the start of a run and the first query
   arrival. The Cloud Online Inference benchmark is meant to measure sustained
   performance. Some inference systems need time to start up. Start up time is
   meant to cover compilation times, booting up accelerators, and other
   once-per-process costs.

4. What is warm up time?

   Warm up time is the time between the first query arrival and the start of
   performance measurement. The Cloud Online Inference benchmark is meant to
   measure sustained performance. Some inference systems need time to warm
   up. Warm up time is meant cover the time necessary to warm up caches, and
   reach steady-state thermal throttling behavior.

5. Why is the QPS of a trace not exactly as requested?

   Trace generation models a Poisson process with a selected QPS generating one
   trace entry at a time until both the minimum trace length and minimum trace
   duration goals at met. When both minimums are met, the effective QPS (number
   of trace entries / timestamp of last trace entry) maybe different from the
   desired average QPS. For sufficiently long traces, the difference should be
   very small.

6. Where do latency bounds come from?

   Model owners set latency bounds to reflect real-world use cases.

7. How can I generate a trace with more or fewer queries than the number of
   unique queries in the trace library?

   Queries are selected from the query library for inclusion in the trace by
   uniform random sampling with replacement.

8. How is an accuracy run different from a performance run?

   In an accuracy run, all queries in the query library are inferred exactly
   once and each inference result is checked for accuracy.

9. How long does a test need to run in order to demonstrate sustained
   performance?

   The MLPerf inference committee is still trying to answer this
   question. Tentatively, we plan to try to determine the minimum duration
   necessary to demonstrate sustained performance empirically. Some proposals
   have suggested that a final verification run of one hour may be
   necessary. Feedback is very welcome!

10. What's a high-percentile latency-bound?

   It's a compromise between attempting to capture worst case tail latency and
   while filtering out the noise at the extreme. Real applications generally use
   a bound somewhere in the 95th to 99th percentile. Lower percentiles yield
   better reproducibility because they are more noise resistant. Since 95th
   percentile is in real world use and is more pragmatic, maybe it is a good
   starting point?

11. Is this tool only for Online Cloud inference, what about Batch Cloud, Online
    Edge and Batch Edge?

   The desire is that this tool can cover all use cases, but Online Cloud is
   being prioritized because it seems to be the most complex and demanding
   case. Feature requests are welcome. Patches are even more welcome!
