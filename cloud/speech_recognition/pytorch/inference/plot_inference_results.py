import os
import sys
import csv
import argparse
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt


def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepSpeech throughput graph plotter')
    parser.add_argument('--results_folder', default=os.getcwd(),
                        help='Location of the exported inference results')
    parser.add_argument('--my_machine', default='Azure E16_v3')
    args = parser.parse_args()

    batch_sizes = [1, 2, 4, 6, 8, 10, 12, 16, 24, 32, 64, 128]
    trials = [""]  # add your trial folders here
    num_warmups = 0

    # Throughput vs Slowdown bound
    fig = plt.figure()
    ax = plt.subplot(111)
    slowdowns99 = dict()
    throughputs = dict()
    for trial in trials:
        for bs in batch_sizes:
            res_path = osp.join(args.results_folder, trial, "inference_bs{}{}.csv".format(bs, "_cpu"))
            with open(res_path, 'r+') as results_file:
                csv_reader = csv.DictReader(results_file)
                prev_row = None
                slowdowns = []
                batch_throughputs = []
                total_audio_processed = 0
                for row in csv_reader:
                    if bs == 1:
                        total_audio_processed += float(row['batch_duration_s'])
                        slowdowns.append(0)
                    else:
                        total_audio_processed += float(row['item_duration_s'])
                        slowdowns.append(1 - float(row['item_latency']) / float(row['batch_latency']))
                    if prev_row is not None and prev_row['batch_num'] != row['batch_num']:
                        batch_throughputs.append(
                            float(prev_row['batch_duration_s']) / float(prev_row['batch_latency']))
                    prev_row = row
                slowdowns99.setdefault(bs, []).append(np.percentile(slowdowns, 99))
                throughputs.setdefault(bs, []).append(sum(batch_throughputs) / float(len(batch_throughputs)))
                sys.stdout.write("\r[{},{}]         ".format(trial, bs))
                sys.stdout.flush()

    slowdowns99_array = [sum(slowdowns99[bs]) / float(len(slowdowns99[bs])) for bs in batch_sizes]
    throughputs_array = [sum(throughputs[bs]) / float(len(throughputs[bs])) for bs in batch_sizes]
    slowdowns99_err_array = [np.std(slowdowns99[bs]) for bs in batch_sizes]
    throughputs_err_array = [np.std(throughputs[bs]) for bs in batch_sizes]
    slowdowns_bound = list(np.linspace(0.0, 1.0, 30))
    slowdowns_bounded_throughputs = []
    slowdowns_bounded_throughputs_xerr = []
    slowdowns_bounded_throughputs_yerr = []
    for bound in slowdowns_bound:
        max_throughput = 0
        xerr = 0
        yerr = 0
        for i, s in enumerate(slowdowns99_array):
            if s <= bound:
                if throughputs_array[i] >= max_throughput:
                    max_throughput = throughputs_array[i]
                    xerr = slowdowns99_err_array[i]
                    yerr = throughputs_err_array[i]
        slowdowns_bounded_throughputs.append(max_throughput)
        slowdowns_bounded_throughputs_xerr.append(xerr)
        slowdowns_bounded_throughputs_yerr.append(yerr)
    ax.plot(slowdowns_bound, slowdowns_bounded_throughputs, '.k-')
    for i, bound in enumerate(slowdowns_bound):
        ax.errorbar(bound, slowdowns_bounded_throughputs[i], xerr=slowdowns_bounded_throughputs_xerr[i],
                    yerr=slowdowns_bounded_throughputs_yerr[i])
    z = np.polyfit(slowdowns_bound[:], slowdowns_bounded_throughputs[:], 6)
    plt.title('Librispeech Test Clean Workload Performance (normalized by batch 1 latency)')
    ax.set_xlabel('Slowdown from batch 1 = 1- (batch 1 latency / batch latency) [sec/sec]')
    ax.set_ylabel('Throughput = audio duration of batch / batch latency [sec/sec]')
    print('Saving plot')
    plt.savefig("slowdown_bounded_througphputs.png")
    print('Section done')

    # Latency

    # 1. Plot the distribution of each sample's runtimes, plot the warmups runs in RED
    # 1b. Compute mean sample runtime and standard deviation (excluding warmups).
    num_warmups = 0
    data = dict()
    fig1 = plt.figure()
    ax = plt.subplot(1, 1, 1)
    runtime_res = []
    for trial in trials:
        color = 'r'
        res_path = osp.join(args.results_folder, trial, "inference_bs{}{}.csv".format(1, "_cpu"))

        with open(res_path, 'r+') as results_file:
            csv_reader = csv.DictReader(results_file)
            for run_num, row in enumerate(csv_reader):
                if run_num >= num_warmups:
                    color = 'b'
                    # Update the data dictionary to compute error bars and print out a detailed summary
                    idx = int(row['batch_num'])
                    latency = float(row['batch_latency'])
                    dur = float(row['batch_duration_s'])
                    runtime_res.append(latency)
                    if not idx in data:
                        data[idx] = {'series': [latency],
                                     'n': 1, 'mean': latency,
                                     'stddev': 0, 'dur': dur}
                    else:
                        prev = data[idx]
                        data[idx]['series'].append(latency)
                        data[idx]['n'] += 1
                        data[idx]['mean'] = sum(data[idx]['series']) / data[idx]['n']
                        data[idx]['stddev'] = np.std(data[idx]['series'])

            ax.plot(idx, latency, marker='o', c=color)

    print('Plot error bars')
    for idx in data:
        ax.errorbar(idx, data[idx]['mean'], yerr=data[idx]['stddev'])
    plt.title(
        '{}: Scatter plot of inference trials and variances of Librispeech Test Clean inputs'.format(args.my_machine))
    plt.xlabel('Idx of input')
    plt.ylabel('Latency of inference trials [sec]')
    print("Saving plot")
    plt.savefig("item_latency.png")
    print('Section done')

    # 1.5 Plot out the normalized plot using 
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    print('Plot realtime speedup')
    for idx in data:
        ax.plot(idx, data[idx]['dur'] / data[idx]['mean'], marker='o', c='g')
    plt.title('{}: Normalized mean performance plot Librispeech Test Clean inputs'.format(args.my_machine))
    plt.xlabel('Idx of input')
    plt.ylabel('Real-time speed up = Input duration / mean Latency of inference trials [sec/sec]')
    print('Saving plot')
    plt.savefig("real_time_speedup.png")
    print('Section done')


    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    print('Plot realtime speedup')
    for idx in data:
        ax.plot(data[idx]['dur'], data[idx]['dur'] / data[idx]['mean'], marker='o', c='g')
    plt.title('{}: Normalized mean performance plot Librispeech Test Clean inputs'.format(args.my_machine))
    plt.xlabel('Input duration')
    plt.ylabel('Real-time speed up = Input duration / mean Latency of inference trials [sec/sec]')
    print('Saving plot')
    plt.savefig("real_time_speedup_vs_duration.png")
    print('Section done')

    # 2. Remove some x warmup runs then plot the CDF
    print('Generating histogram')
    hist, bin_edges = np.histogram(runtime_res, bins=70)
    print('Generating cdf')
    cdf = np.cumsum(hist)
    print('Ploting cdf')
    fig2 = plt.figure(2)
    ax2 = plt.subplot(1, 1, 1)
    ax2.plot(bin_edges[1:], cdf)
    plt.xlabel('Latency bound [sec]')
    plt.ylabel('% of samples')
    plt.title(
        '{}: % of batch size 1 inputs from Librispeech Test Clean satisfying a latency bound'.format(args.my_machine))
    plt.xticks(bin_edges, rotation=90)
    plt.yticks(cdf[::10], np.round(cdf / cdf[-1], 2)[::10])
    plt.axhline(y=0.99 * cdf[-1], xmin=0, xmax=bin_edges[-1], c='k')
    plt.axvline(x=bin_edges[find_nearest_idx(cdf / cdf[-1], 0.99)], ymin=0, ymax=1, c='k')
    print('Saving plot')
    plt.savefig("cdf.png")
    print('Section done')
