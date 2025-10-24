#!/usr/bin/env python3
import re
import matplotlib.pyplot as plt
from datetime import datetime
import sys

def parse_log_file(log_file_path):
    """Parse the log file to extract throughput data"""
    throughput_data = []
    
    with open(log_file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            # Look for lines with throughput information
            if 'Avg prompt throughput:' in line and 'Avg generation throughput:' in line:
                # Extract timestamp
                timestamp_match = re.search(r'(\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                if timestamp_match:
                    timestamp_str = timestamp_match.group(1)
                    # Parse timestamp (assuming current year)
                    timestamp = datetime.strptime(f"2024-{timestamp_str}", "%Y-%m-%d %H:%M:%S")
                    
                    # Extract prompt throughput
                    prompt_match = re.search(r'Avg prompt throughput: ([\d.]+) tokens/s', line)
                    prompt_throughput = float(prompt_match.group(1)) if prompt_match else 0.0
                    
                    # Extract generation throughput
                    gen_match = re.search(r'Avg generation throughput: ([\d.]+) tokens/s', line)
                    gen_throughput = float(gen_match.group(1)) if gen_match else 0.0
                    
                    # Extract running requests
                    running_match = re.search(r'Running: (\d+) reqs', line)
                    running_reqs = int(running_match.group(1)) if running_match else 0
                    
                    # Extract waiting requests
                    waiting_match = re.search(r'Waiting: (\d+) reqs', line)
                    waiting_reqs = int(waiting_match.group(1)) if waiting_match else 0
                    
                    # Extract GPU KV cache usage
                    cache_match = re.search(r'GPU KV cache usage: ([\d.]+)%', line)
                    cache_usage = float(cache_match.group(1)) if cache_match else 0.0
                    
                    throughput_data.append({
                        'timestamp': timestamp,
                        'prompt_throughput': prompt_throughput,
                        'generation_throughput': gen_throughput,
                        'running_reqs': running_reqs,
                        'waiting_reqs': waiting_reqs,
                        'cache_usage': cache_usage,
                        'line_number': line_num
                    })
    
    return throughput_data

def create_simple_plots(data, filename="Unknown", no_zeros=False):
    """Create simple line plots for throughput data"""
    if not data:
        print("No throughput data found!")
        return
    
    # Extract data for plotting
    timestamps = [d['timestamp'] for d in data]
    prompt_throughput = [d['prompt_throughput'] for d in data]
    gen_throughput = [d['generation_throughput'] for d in data]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle(f'vLLM Engine Throughput Analysis - {filename}', fontsize=16, fontweight='bold')
    
    # Plot 1: Generation Throughput
    ax1.plot(timestamps, gen_throughput, 'b-', linewidth=2, marker='o', markersize=4, label='Generation Throughput')
    ax1.set_title('Generation Throughput Over Time', fontweight='bold', fontsize=14)
    ax1.set_ylabel('Tokens/sec', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    
    # Add data values as text annotations (every 5th point to avoid clutter)
    for i in range(0, len(timestamps), 5):
        ax1.annotate(f'{gen_throughput[i]:.1f}', 
                    (timestamps[i], gen_throughput[i]), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center', 
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    # Plot 2: Prompt Throughput
    ax2.plot(timestamps, prompt_throughput, 'r-', linewidth=2, marker='s', markersize=4, label='Prompt Throughput')
    ax2.set_title('Prompt Throughput Over Time', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Tokens/sec', fontsize=12)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)
    
    # Add data values as text annotations (every 5th point to avoid clutter)
    for i in range(0, len(timestamps), 5):
        ax2.annotate(f'{prompt_throughput[i]:.1f}', 
                    (timestamps[i], prompt_throughput[i]), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center', 
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    # Rotate x-axis labels for better readability
    for ax in [ax1, ax2]:
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='both', labelsize=10)
    
    plt.tight_layout()
    output_filename = f'/root/simple_throughput_analysis_{filename.replace("/", "_").replace(".", "_")}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    def calculate_median(data):
        sorted_data = sorted(data)
        n = len(sorted_data)
        if n % 2 == 0:
            return (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
        else:
            return sorted_data[n//2]
    
    # Filter out zeros if requested
    if no_zeros:
        gen_throughput_filtered = [x for x in gen_throughput if x > 0]
        prompt_throughput_filtered = [x for x in prompt_throughput if x > 0]
        print(f"  (Excluding zeros: {len(gen_throughput) - len(gen_throughput_filtered)} gen zeros, {len(prompt_throughput) - len(prompt_throughput_filtered)} prompt zeros)")
    else:
        gen_throughput_filtered = gen_throughput
        prompt_throughput_filtered = prompt_throughput
    
    if gen_throughput_filtered:
        gen_avg = sum(gen_throughput_filtered)/len(gen_throughput_filtered)
        gen_max = max(gen_throughput_filtered)
        gen_min = min(gen_throughput_filtered)
        gen_median = calculate_median(gen_throughput_filtered)
    else:
        gen_avg = gen_max = gen_min = gen_median = 0
    
    if prompt_throughput_filtered:
        prompt_avg = sum(prompt_throughput_filtered)/len(prompt_throughput_filtered)
        prompt_max = max(prompt_throughput_filtered)
        prompt_min = min(prompt_throughput_filtered)
        prompt_median = calculate_median(prompt_throughput_filtered)
    else:
        prompt_avg = prompt_max = prompt_min = prompt_median = 0
    
    print(f"\n=== THROUGHPUT SUMMARY STATISTICS - {filename} ===")
    print(f"Total data points: {len(data)}")
    print(f"Time range: {timestamps[0]} to {timestamps[-1]}")
    
    print(f"\nGeneration Throughput:")
    print(f"  Average: {gen_avg:.2f} tokens/sec")
    print(f"  Minimum: {gen_min:.2f} tokens/sec")
    print(f"  Maximum: {gen_max:.2f} tokens/sec")
    print(f"  Median: {gen_median:.2f} tokens/sec")
    
    print(f"\nPrompt Throughput:")
    print(f"  Average: {prompt_avg:.2f} tokens/sec")
    print(f"  Minimum: {prompt_min:.2f} tokens/sec")
    print(f"  Maximum: {prompt_max:.2f} tokens/sec")
    print(f"  Median: {prompt_median:.2f} tokens/sec")
    
    return {
        'filename': filename,
        'data_points': len(data),
        'gen_avg': sum(gen_throughput)/len(gen_throughput),
        'gen_max': max(gen_throughput),
        'gen_min': min(gen_throughput),
        'prompt_avg': sum(prompt_throughput)/len(prompt_throughput),
        'prompt_max': max(prompt_throughput),
        'prompt_min': min(prompt_throughput)
    }

def create_comparison_plots(data1, data2, filename1, filename2, no_zeros=False):
    """Create comparison plots for two log files using sample numbers as x-axis"""
    if not data1 or not data2:
        print("Cannot create comparison plots - missing data!")
        return
    
    # Extract data for both files
    prompt_throughput1 = [d['prompt_throughput'] for d in data1]
    gen_throughput1 = [d['generation_throughput'] for d in data1]
    cache_usage1 = [d['cache_usage'] for d in data1]
    running_reqs1 = [d['running_reqs'] for d in data1]
    waiting_reqs1 = [d['waiting_reqs'] for d in data1]
    
    prompt_throughput2 = [d['prompt_throughput'] for d in data2]
    gen_throughput2 = [d['generation_throughput'] for d in data2]
    cache_usage2 = [d['cache_usage'] for d in data2]
    running_reqs2 = [d['running_reqs'] for d in data2]
    waiting_reqs2 = [d['waiting_reqs'] for d in data2]
    
    # Create sample numbers (x-axis)
    samples1 = list(range(1, len(data1) + 1))
    samples2 = list(range(1, len(data2) + 1))
    
    # Create figure with comparison plots
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('vLLM Engine Complete Performance Comparison (Sample-based)', fontsize=16, fontweight='bold')
    
    # Plot 1: Generation Throughput Comparison
    ax1.plot(samples1, gen_throughput1, 'b-', linewidth=2, marker='o', markersize=4, label=f'File 1: {filename1.split("/")[-1]}')
    ax1.plot(samples2, gen_throughput2, 'g-', linewidth=2, marker='s', markersize=4, label=f'File 2: {filename2.split("/")[-1]}')
    ax1.set_title('Generation Throughput Comparison', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Sample Number', fontsize=12)
    ax1.set_ylabel('Tokens/sec', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    
    # Add data values as text annotations (every 5th point to avoid clutter)
    for i in range(0, len(samples1), 5):
        ax1.annotate(f'{gen_throughput1[i]:.1f}', 
                    (samples1[i], gen_throughput1[i]), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center', 
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    for i in range(0, len(samples2), 5):
        ax1.annotate(f'{gen_throughput2[i]:.1f}', 
                    (samples2[i], gen_throughput2[i]), 
                    textcoords="offset points", 
                    xytext=(0,-15), 
                    ha='center', 
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    # Plot 2: Prompt Throughput Comparison
    ax2.plot(samples1, prompt_throughput1, 'r-', linewidth=2, marker='o', markersize=4, label=f'File 1: {filename1.split("/")[-1]}')
    ax2.plot(samples2, prompt_throughput2, 'orange', linewidth=2, marker='s', markersize=4, label=f'File 2: {filename2.split("/")[-1]}')
    ax2.set_title('Prompt Throughput Comparison', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Tokens/sec', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)
    
    # Add data values as text annotations (every 5th point to avoid clutter)
    for i in range(0, len(samples1), 5):
        ax2.annotate(f'{prompt_throughput1[i]:.1f}', 
                    (samples1[i], prompt_throughput1[i]), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center', 
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    for i in range(0, len(samples2), 5):
        ax2.annotate(f'{prompt_throughput2[i]:.1f}', 
                    (samples2[i], prompt_throughput2[i]), 
                    textcoords="offset points", 
                    xytext=(0,-15), 
                    ha='center', 
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="moccasin", alpha=0.7))
    
    # Plot 3: KV Cache Usage Comparison
    ax3.plot(samples1, cache_usage1, 'purple', linewidth=2, marker='o', markersize=4, label=f'File 1: {filename1.split("/")[-1]}')
    ax3.plot(samples2, cache_usage2, 'brown', linewidth=2, marker='s', markersize=4, label=f'File 2: {filename2.split("/")[-1]}')
    ax3.set_title('GPU KV Cache Usage Comparison', fontweight='bold', fontsize=14)
    ax3.set_xlabel('Sample Number', fontsize=12)
    ax3.set_ylabel('Cache Usage (%)', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=12)
    
    # Add data values as text annotations (every 5th point to avoid clutter)
    for i in range(0, len(samples1), 5):
        ax3.annotate(f'{cache_usage1[i]:.1f}%', 
                    (samples1[i], cache_usage1[i]), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center', 
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="plum", alpha=0.7))
    
    for i in range(0, len(samples2), 5):
        ax3.annotate(f'{cache_usage2[i]:.1f}%', 
                    (samples2[i], cache_usage2[i]), 
                    textcoords="offset points", 
                    xytext=(0,-15), 
                    ha='center', 
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))
    
    # Plot 4: Running Requests Comparison
    ax4.plot(samples1, running_reqs1, 'darkgreen', linewidth=2, marker='o', markersize=3, label=f'File 1: {filename1.split("/")[-1]}')
    ax4.plot(samples2, running_reqs2, 'lime', linewidth=2, marker='s', markersize=3, label=f'File 2: {filename2.split("/")[-1]}')
    ax4.set_title('Running Requests', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Number of Requests', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)
    
    # Plot 5: Waiting Requests Comparison
    ax5.plot(samples1, waiting_reqs1, 'darkred', linewidth=2, marker='o', markersize=3, label=f'File 1: {filename1.split("/")[-1]}')
    ax5.plot(samples2, waiting_reqs2, 'pink', linewidth=2, marker='s', markersize=3, label=f'File 2: {filename2.split("/")[-1]}')
    ax5.set_title('Waiting Requests', fontweight='bold', fontsize=12)
    ax5.set_xlabel('Sample Number', fontsize=10)
    ax5.set_ylabel('Number of Requests', fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=9)
    
    # Plot 6: Total Requests (Running + Waiting)
    total_reqs1 = [r + w for r, w in zip(running_reqs1, waiting_reqs1)]
    total_reqs2 = [r + w for r, w in zip(running_reqs2, waiting_reqs2)]
    ax6.plot(samples1, total_reqs1, 'navy', linewidth=2, marker='o', markersize=3, label=f'File 1: {filename1.split("/")[-1]}')
    ax6.plot(samples2, total_reqs2, 'cyan', linewidth=2, marker='s', markersize=3, label=f'File 2: {filename2.split("/")[-1]}')
    ax6.set_title('Total Requests (Running + Waiting)', fontweight='bold', fontsize=12)
    ax6.set_xlabel('Sample Number', fontsize=10)
    ax6.set_ylabel('Total Requests', fontsize=10)
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=9)
    
    # Set tick parameters
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.tick_params(axis='both', labelsize=9)
    
    plt.tight_layout()
    comparison_filename = f'./throughput_comparison_{filename1.split("/")[-1].replace(".", "_")}_vs_{filename2.split("/")[-1].replace(".", "_")}.png'
    plt.savefig(comparison_filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate statistics
    def calculate_median(data):
        sorted_data = sorted(data)
        n = len(sorted_data)
        if n % 2 == 0:
            return (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
        else:
            return sorted_data[n//2]
    
    # Filter out zeros if requested
    if no_zeros:
        gen_throughput1_filtered = [x for x in gen_throughput1 if x > 0]
        gen_throughput2_filtered = [x for x in gen_throughput2 if x > 0]
        prompt_throughput1_filtered = [x for x in prompt_throughput1 if x > 0]
        prompt_throughput2_filtered = [x for x in prompt_throughput2 if x > 0]
        cache_usage1_filtered = [x for x in cache_usage1 if x > 0]
        cache_usage2_filtered = [x for x in cache_usage2 if x > 0]
        print(f"  (Excluding zeros: {len(gen_throughput1) - len(gen_throughput1_filtered)} gen1, {len(gen_throughput2) - len(gen_throughput2_filtered)} gen2)")
    else:
        gen_throughput1_filtered = gen_throughput1
        gen_throughput2_filtered = gen_throughput2
        prompt_throughput1_filtered = prompt_throughput1
        prompt_throughput2_filtered = prompt_throughput2
        cache_usage1_filtered = cache_usage1
        cache_usage2_filtered = cache_usage2
    
    # Calculate generation throughput statistics
    if gen_throughput1_filtered:
        gen_avg1 = sum(gen_throughput1_filtered)/len(gen_throughput1_filtered)
        gen_max1 = max(gen_throughput1_filtered)
        gen_min1 = min(gen_throughput1_filtered)
        gen_median1 = calculate_median(gen_throughput1_filtered)
    else:
        gen_avg1 = gen_max1 = gen_min1 = gen_median1 = 0
        
    if gen_throughput2_filtered:
        gen_avg2 = sum(gen_throughput2_filtered)/len(gen_throughput2_filtered)
        gen_max2 = max(gen_throughput2_filtered)
        gen_min2 = min(gen_throughput2_filtered)
        gen_median2 = calculate_median(gen_throughput2_filtered)
    else:
        gen_avg2 = gen_max2 = gen_min2 = gen_median2 = 0
    
    # Calculate prompt throughput statistics
    if prompt_throughput1_filtered:
        prompt_avg1 = sum(prompt_throughput1_filtered)/len(prompt_throughput1_filtered)
        prompt_max1 = max(prompt_throughput1_filtered)
        prompt_min1 = min(prompt_throughput1_filtered)
        prompt_median1 = calculate_median(prompt_throughput1_filtered)
    else:
        prompt_avg1 = prompt_max1 = prompt_min1 = prompt_median1 = 0
        
    if prompt_throughput2_filtered:
        prompt_avg2 = sum(prompt_throughput2_filtered)/len(prompt_throughput2_filtered)
        prompt_max2 = max(prompt_throughput2_filtered)
        prompt_min2 = min(prompt_throughput2_filtered)
        prompt_median2 = calculate_median(prompt_throughput2_filtered)
    else:
        prompt_avg2 = prompt_max2 = prompt_min2 = prompt_median2 = 0
    
    # Calculate cache usage statistics
    if cache_usage1_filtered:
        cache_avg1 = sum(cache_usage1_filtered)/len(cache_usage1_filtered)
        cache_max1 = max(cache_usage1_filtered)
        cache_min1 = min(cache_usage1_filtered)
        cache_median1 = calculate_median(cache_usage1_filtered)
    else:
        cache_avg1 = cache_max1 = cache_min1 = cache_median1 = 0
        
    if cache_usage2_filtered:
        cache_avg2 = sum(cache_usage2_filtered)/len(cache_usage2_filtered)
        cache_max2 = max(cache_usage2_filtered)
        cache_min2 = min(cache_usage2_filtered)
        cache_median2 = calculate_median(cache_usage2_filtered)
    else:
        cache_avg2 = cache_max2 = cache_min2 = cache_median2 = 0
    
    print(f"\n=== COMPARISON SUMMARY ===")
    print(f"File 1 ({filename1}): {len(data1)} data points")
    print(f"File 2 ({filename2}): {len(data2)} data points")
    print(f"\nGeneration Throughput Comparison:")
    print(f"  File 1 - Avg: {gen_avg1:.2f}, Min: {gen_min1:.2f}, Max: {gen_max1:.2f}, Median: {gen_median1:.2f}")
    print(f"  File 2 - Avg: {gen_avg2:.2f}, Min: {gen_min2:.2f}, Max: {gen_max2:.2f}, Median: {gen_median2:.2f}")
    print(f"  Difference - Avg: {gen_avg2-gen_avg1:+.2f}, Min: {gen_min2-gen_min1:+.2f}, Max: {gen_max2-gen_max1:+.2f}, Median: {gen_median2-gen_median1:+.2f}")
    print(f"\nPrompt Throughput Comparison:")
    print(f"  File 1 - Avg: {prompt_avg1:.2f}, Min: {prompt_min1:.2f}, Max: {prompt_max1:.2f}, Median: {prompt_median1:.2f}")
    print(f"  File 2 - Avg: {prompt_avg2:.2f}, Min: {prompt_min2:.2f}, Max: {prompt_max2:.2f}, Median: {prompt_median2:.2f}")
    print(f"  Difference - Avg: {prompt_avg2-prompt_avg1:+.2f}, Min: {prompt_min2-prompt_min1:+.2f}, Max: {prompt_max2-prompt_max1:+.2f}, Median: {prompt_median2-prompt_median1:+.2f}")
    print(f"\nKV Cache Usage Comparison:")
    print(f"  File 1 - Avg: {cache_avg1:.2f}%, Min: {cache_min1:.2f}%, Max: {cache_max1:.2f}%, Median: {cache_median1:.2f}%")
    print(f"  File 2 - Avg: {cache_avg2:.2f}%, Min: {cache_min2:.2f}%, Max: {cache_max2:.2f}%, Median: {cache_median2:.2f}%")
    print(f"  Difference - Avg: {cache_avg2-cache_avg1:+.2f}%, Min: {cache_min2-cache_min1:+.2f}%, Max: {cache_max2-cache_max1:+.2f}%, Median: {cache_median2-cache_median1:+.2f}%")
    
    # Calculate request statistics (requests don't need zero filtering as they're counts)
    running_avg1 = sum(running_reqs1)/len(running_reqs1)
    running_avg2 = sum(running_reqs2)/len(running_reqs2)
    running_max1 = max(running_reqs1)
    running_max2 = max(running_reqs2)
    running_min1 = min(running_reqs1)
    running_min2 = min(running_reqs2)
    running_median1 = calculate_median(running_reqs1)
    running_median2 = calculate_median(running_reqs2)
    
    waiting_avg1 = sum(waiting_reqs1)/len(waiting_reqs1)
    waiting_avg2 = sum(waiting_reqs2)/len(waiting_reqs2)
    waiting_max1 = max(waiting_reqs1)
    waiting_max2 = max(waiting_reqs2)
    waiting_min1 = min(waiting_reqs1)
    waiting_min2 = min(waiting_reqs2)
    waiting_median1 = calculate_median(waiting_reqs1)
    waiting_median2 = calculate_median(waiting_reqs2)
    
    print(f"\nRunning Requests Comparison:")
    print(f"  File 1 - Avg: {running_avg1:.1f}, Min: {running_min1}, Max: {running_max1}, Median: {running_median1:.1f}")
    print(f"  File 2 - Avg: {running_avg2:.1f}, Min: {running_min2}, Max: {running_max2}, Median: {running_median2:.1f}")
    print(f"  Difference - Avg: {running_avg2-running_avg1:+.1f}, Min: {running_min2-running_min1:+d}, Max: {running_max2-running_max1:+d}, Median: {running_median2-running_median1:+.1f}")
    
    print(f"\nWaiting Requests Comparison:")
    print(f"  File 1 - Avg: {waiting_avg1:.1f}, Min: {waiting_min1}, Max: {waiting_max1}, Median: {waiting_median1:.1f}")
    print(f"  File 2 - Avg: {waiting_avg2:.1f}, Min: {waiting_min2}, Max: {waiting_max2}, Median: {waiting_median2:.1f}")
    print(f"  Difference - Avg: {waiting_avg2-waiting_avg1:+.1f}, Min: {waiting_min2-waiting_min1:+d}, Max: {waiting_max2-waiting_max1:+d}, Median: {waiting_median2-waiting_median1:+.1f}")
    
    return comparison_filename

if __name__ == "__main__":
    # Check if file path(s) are provided as command line arguments
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single file: python3 simple_throughput_plot.py <log_file_path> [--no-zeros] [--tag <custom_tag>]")
        print("  Two files:   python3 simple_throughput_plot.py <log_file1> <log_file2> [--no-zeros] [--tag1 <tag1>] [--tag2 <tag2>]")
        print("Options:")
        print("  --no-zeros    Exclude zero values from statistics calculations")
        print("  --tag <tag>   Custom tag for single file (replaces filename in plots)")
        print("  --tag1 <tag>  Custom tag for first file in comparison")
        print("  --tag2 <tag>  Custom tag for second file in comparison")
        print("Examples:")
        print("  python3 simple_throughput_plot.py /root/qwen_tp4_3.2.2.log --tag 'Baseline'")
        print("  python3 simple_throughput_plot.py /root/log1.log /root/log2.log --tag1 'Experiment A' --tag2 'Experiment B' --no-zeros")
        sys.exit(1)
    
    # Parse command line arguments
    args = sys.argv[1:]
    log_files = []
    tags = []
    no_zeros = False
    
    i = 0
    while i < len(args):
        if args[i] == '--no-zeros':
            no_zeros = True
        elif args[i] == '--tag' and i + 1 < len(args):
            tags.append(args[i + 1])
            i += 1
        elif args[i] == '--tag1' and i + 1 < len(args):
            tags.append(args[i + 1])
            i += 1
        elif args[i] == '--tag2' and i + 1 < len(args):
            tags.append(args[i + 1])
            i += 1
        elif not args[i].startswith('--'):
            log_files.append(args[i])
        i += 1
    
    # Determine mode and process
    if len(log_files) == 1:
        # Single file mode
        log_file = log_files[0]
        tag = tags[0] if tags else log_file.split('/')[-1]
        
        print(f"Parsing log file: {log_file}")
        print(f"Using tag: {tag}")
        if no_zeros:
            print("Excluding zero values from statistics")
        
        data = parse_log_file(log_file)
        
        if data:
            print(f"Found {len(data)} throughput data points")
            create_simple_plots(data, tag, no_zeros)
        else:
            print("No throughput data found in the log file!")
    
    elif len(log_files) == 2:
        # Two file comparison mode
        log_file1, log_file2 = log_files
        tag1 = tags[0] if len(tags) > 0 else log_file1.split('/')[-1]
        tag2 = tags[1] if len(tags) > 1 else log_file2.split('/')[-1]
        
        print(f"Parsing first log file: {log_file1}")
        print(f"Using tag: {tag1}")
        print(f"Parsing second log file: {log_file2}")
        print(f"Using tag: {tag2}")
        if no_zeros:
            print("Excluding zero values from statistics")
        
        data1 = parse_log_file(log_file1)
        data2 = parse_log_file(log_file2)
        
        if data1 and data2:
            print(f"Found {len(data1)} data points in first file")
            print(f"Found {len(data2)} data points in second file")
            
            # Create individual plots
            print("\nCreating individual plots...")
            stats1 = create_simple_plots(data1, tag1, no_zeros)
            stats2 = create_simple_plots(data2, tag2, no_zeros)
            
            # Create comparison plots
            print("\nCreating comparison plots...")
            comparison_file = create_comparison_plots(data1, data2, tag1, tag2, no_zeros)
            print(f"Comparison plots saved to: {comparison_file}")
            
        else:
            print("Cannot create comparison - missing data in one or both files!")
            if not data1:
                print(f"No throughput data found in: {log_file1}")
            if not data2:
                print(f"No throughput data found in: {log_file2}")
    
    else:
        print("Please provide 1 or 2 log file paths.")
        sys.exit(1)
