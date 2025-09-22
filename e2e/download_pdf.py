#!/usr/bin/env python3
import pandas as pd
import os
import argparse
import subprocess
import ast
from pathlib import Path
from multiprocessing import Pool, cpu_count
import time
import sys
from tqdm import tqdm
from datasets import load_dataset
import getpass
import re

def download_frames_dataset(output_dir):
    """Download the FRAMES dataset from Hugging Face and save as TSV."""
    print("Downloading FRAMES dataset from Hugging Face...")
    
    try:
        # Load the dataset
        dataset = load_dataset("google/frames-benchmark", split="test")
        
        # Convert to pandas DataFrame
        df = dataset.to_pandas()
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save as TSV
        tsv_path = output_dir / "frames_dataset.tsv"
        df.to_csv(tsv_path, sep='\t', index=False)
        
        print(f"✅ FRAMES dataset downloaded and saved to: {tsv_path}")
        print(f"Dataset contains {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
        
        return str(tsv_path)
        
    except Exception as e:
        print(f"❌ Error downloading FRAMES dataset: {e}")
        return None

def retry_failed_urls(failed_urls, output_dir, processes):
    """Retry downloading failed URLs."""
    if not failed_urls:
        return
    
    print(f"\n=== RETRYING FAILED DOWNLOADS ===")
    print(f"Retrying {len(failed_urls)} failed URLs...")
    
    # Prepare arguments for retry
    retry_args = [(url, output_dir, i+1, len(failed_urls)) for i, (_, _, url) in enumerate(failed_urls)]
    
    # Create progress bar for retry
    progress_bar = tqdm(total=len(failed_urls), desc="Retrying...", unit="URL")
    
    successful_retries = 0
    still_failed = []
    
    with Pool(processes=processes) as pool:
        for result in pool.imap(process_url, retry_args):
            success, filename, status, url = result
            
            if success:
                progress_bar.set_description(f"✓ Retry Success: {filename[:30]}...")
                successful_retries += 1
            else:
                progress_bar.set_description(f"✗ Retry Failed: {filename[:30]}...")
                still_failed.append((filename, status, url))
                print(f"\n❌ RETRY FAILED: {filename}")
                print(f"   URL: {url}")
                print(f"   Error: {status}")
            
            progress_bar.update(1)
    
    progress_bar.close()
    
    print(f"\nRetry complete! Successfully retried: {successful_retries}, Still failed: {len(still_failed)}")
    
    if still_failed:
        print(f"\n=== STILL FAILED AFTER RETRY ===")
        for i, (filename, status, url) in enumerate(still_failed, 1):
            print(f"{i:2d}. {filename}")
            print(f"    URL: {url}")
            print(f"    Error: {status}")
            print()

def process_url(args_tuple):
    """Process a single URL - designed for multiprocessing."""
    url, output_dir, index, total = args_tuple
    
    # Create safe filename
    filename = url.replace("https://", "").replace("/", "_").replace(":", "_")
    if len(filename) > 200:
        filename = filename[:200]
    filename += ".pdf"
    
    output_path = output_dir / filename
    
    # Skip if file already exists
    if output_path.exists():
        return True, filename, "Skipping", url
    
    # Run wkhtmltopdf command
    command = f'wkhtmltopdf --page-size A4 --margin-top 0.75in --margin-right 0.75in --margin-bottom 0.75in --margin-left 0.75in --encoding UTF-8 --load-error-handling ignore --load-media-error-handling ignore --javascript-delay 2000 "{url}" "{output_path}"'
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            # Check if file was actually created and has reasonable size
            if output_path.exists() and output_path.stat().st_size > 1000:
                return True, filename, "Success", url
            else:
                error_msg = f"File too small or empty ({output_path.stat().st_size if output_path.exists() else 0} bytes)"
                return False, filename, error_msg, url
        else:
            error_msg = f"wkhtmltopdf failed (return code: {result.returncode})"
            if result.stderr:
                error_msg += f" - {result.stderr[:100]}"
            return False, filename, error_msg, url
    except subprocess.TimeoutExpired:
        return False, filename, "Timeout (120s)", url
    except Exception as e:
        return False, filename, f"Exception: {str(e)[:100]}", url


def main():
    parser = argparse.ArgumentParser(description='Download FRAMES dataset from Hugging Face and convert URLs to PDFs')
    parser.add_argument('--tsv_path', default=None, help='Input TSV file (default: download FRAMES dataset)')
    parser.add_argument('--max_urls', type=int, default=None, help='Maximum number of URLs to process (default: all)')
    parser.add_argument('--output_pdf', default='doc_pdf', help='Output directory for PDFs (default: doc_pdf)')
    parser.add_argument('--output_data', default='data', help='Output directory for dataset file, if downloaded from Hugging Face (default: data)')
    parser.add_argument('--processes', type=int, default=10, help='Number of parallel processes (default: 10)')
    
    args = parser.parse_args()

    # Set XDG_RUNTIME_DIR for wkhtmltopdf
    user = getpass.getuser()
    xdg_runtime_dir = f"/tmp/runtime-{user}"
    os.environ["XDG_RUNTIME_DIR"] = xdg_runtime_dir
    
    # Create output directories
    output_dir = Path(args.output_pdf)
    output_dir.mkdir(exist_ok=True)
    
    # Determine TSV file path
    if args.tsv_path is None:
        print("=== DOWNLOADING FRAMES DATASET ===")
        tsv_path = download_frames_dataset(args.output_data)
        if tsv_path is None:
            print("❌ Failed to download FRAMES dataset. Exiting.")
            return
        args.tsv_path = tsv_path
    else:
        print(f"Using provided TSV file: {args.tsv_path}")

    # Load dataset
    df = pd.read_csv(args.tsv_path, sep='\t')

    # df['wiki_links'] = df['wiki_links'].apply(ast.literal_eval)
    # wiki_dict = df['wiki_links'].to_dict()

    def extract_wikipedia_links(item):
        # List may itself have a single string with multiple links, comma separated
        # Hence, we use regex matching 

        # Convert string to proper list if needed
        if isinstance(item, str):
            item = ast.literal_eval(item)

        # Results holder
        links = []
        for entry in item:
            # Find all links, including in Markdown format, within the string
            matches = re.findall(r'https://en\.wikipedia\.org/wiki/[^\s,\]]+', entry)
            links.extend(matches)
        return links

    # Apply to all rows in df.wiki_links
    urls = set()
    for item in df.wiki_links:
        for link in extract_wikipedia_links(item):
            urls.add(link)

    urls = list(urls)
    if args.max_urls:
        urls = urls[:args.max_urls]
    
    if not urls:
        print("No URLs found to process")
        return
    
    print(f"Processing {len(urls)} URLs with {args.processes} parallel processes...")
    
    # Create progress bar
    progress_bar = tqdm(total=len(urls), desc="Starting downloads...", unit="URL")
    
    # Process URLs in parallel with progress bar
    start_time = time.time()
    
    # Prepare arguments for multiprocessing
    process_args = [(url, output_dir, i+1, len(urls)) for i, url in enumerate(urls)]
    
    # Process with progress bar updates
    results = []
    failed_urls = []  # Track failed URLs for detailed reporting
    
    with Pool(processes=args.processes) as pool:
        for result in pool.imap(process_url, process_args):
            success, filename, status, url = result
            results.append((success, filename))
            
            # Update progress bar with status
            if status == "Skipping":
                progress_bar.set_description(f"Skipping: {filename[:30]}...")
            elif status == "Success":
                progress_bar.set_description(f"✓ Success: {filename[:30]}...")
            else:
                # This is a failure case
                progress_bar.set_description(f"✗ {status}: {filename[:30]}...")
                failed_urls.append((filename, status, url))
                print(f"\n❌ FAILED: {filename}")
                print(f"   URL: {url}")
                print(f"   Error: {status}")
            
            progress_bar.update(1)
    
    progress_bar.close()
    
    # Count results
    successful = sum(1 for success, _ in results if success)
    failed = len(results) - successful
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nConversion complete! Successful: {successful}, Failed: {failed}")
    print(f"Total time: {duration:.2f} seconds")
    print(f"Average time per URL: {duration/len(urls):.2f} seconds")
    
    # Print detailed failure report if there were failures
    if failed_urls:
        print(f"\n=== FAILED DOWNLOADS DETAILS ===")
        for i, (filename, status, url) in enumerate(failed_urls, 1):
            print(f"{i:2d}. {filename}")
            print(f"    URL: {url}")
            print(f"    Error: {status}")
            print()
        
        # Ask user if they want to retry failed URLs
        retry_failed_urls(failed_urls, output_dir, args.processes)
    else:
        print(f"\n✅ All downloads completed successfully!")

if __name__ == "__main__":
    main()

