# Copyright 2025 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================


#!/usr/bin/env python3
"""
Download documents (PDF/HTML) from Wikipedia URLs with enhanced error handling.

This script provides a unified interface for downloading Wikipedia pages as either PDF or HTML files,
with robust error handling, retry logic, and concurrent processing.
"""

import asyncio
import json
import argparse
import random
import time
import concurrent.futures
from pathlib import Path
from typing import List, Set, Tuple, Dict, Optional, Any
from abc import ABC, abstractmethod
import re
import urllib.parse
import logging
from collections import Counter
from utils import save_url_mapping, get_base_filename
import subprocess
import os
import getpass
import ast
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset

try:
    import requests
except ImportError:
    requests = None


def fix_malformed_url(url: str) -> str:
    # Remove text fragments (everything after #:~:text=)
    if '#:~:text=' in url:
        url = url.split('#:~:text=')[0]

    # Fix double URL-encoding. The FRAMES dataset occasionally contains URLs
    # whose percent signs were themselves percent-encoded, e.g. "%2527" should
    # be "%27" (an apostrophe). Decode one extra layer when we see "%25" sequences
    # followed by two hex digits.
    if re.search(r'%25[0-9A-Fa-f]{2}', url):
        url = re.sub(
            r'%25([0-9A-Fa-f]{2})',
            lambda m: '%' + m.group(1),
            url,
        )

    # Fix missing closing parentheses
    open_parens = url.count('(')
    close_parens = url.count(')')

    if open_parens > close_parens:
        missing_closes = open_parens - close_parens
        url += ')' * missing_closes

    return url





class BaseDownloader(ABC):
    """Base class for downloading web pages in different formats."""
    
    def __init__(self, output_dir: str, processes: int = 10):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.processes = processes
        self.url_mapping = {}
    
    @abstractmethod
    def get_file_extension(self) -> str:
        """Return the file extension for this downloader."""
        pass
    
    def create_filename(self, url: str) -> str:
        """Generate a filename from URL."""
        filename = url.replace("https://", "").replace("/", "_").replace(":", "_")
        
        # Truncate if too long
        max_length = 200
        if len(filename) > max_length:
            filename = filename[:max_length]
        
        return filename + self.get_file_extension()
    
    @abstractmethod
    def download_single_url(self, url: str, output_path: Path) -> Tuple[bool, str]:
        """
        Download a single URL to the specified path.
        
        Returns:
            Tuple of (success: bool, error_message: str)
        """
        pass
    
    def process_url(self, args_tuple: Tuple[str, Path, int, int]) -> Tuple[bool, str, str, str]:
        """Process a single URL - designed for multiprocessing."""
        url, output_dir, index, total = args_tuple
        
        # Create safe filename
        filename = self.create_filename(url)
        output_path = output_dir / filename
        
        # Skip if file already exists
        if output_path.exists():
            return True, filename, "Skipping", url
        
        # Download the URL
        try:
            success, error_msg = self.download_single_url(url, output_path)
            if success:
                # Check if file was actually created and has reasonable size
                if output_path.exists() and output_path.stat().st_size > 100:
                    return True, filename, "Success", url
                else:
                    size = output_path.stat().st_size if output_path.exists() else 0
                    error_msg = f"File too small or empty ({size} bytes)"
                    return False, filename, error_msg, url
            else:
                return False, filename, error_msg, url
                
        except Exception as e:
            return False, filename, f"Exception: {str(e)[:100]}", url
    
    def download_urls(self, urls: List[str], retry_failures: bool = True) -> Dict[str, Any]:
        """Download multiple URLs with parallel processing and progress tracking."""
        
        if not urls:
            print("No URLs found to process")
            return {"successful": 0, "failed": 0, "failed_urls": []}
        
        print(f"Processing {len(urls)} URLs with {self.processes} parallel processes...")
        
        # Create progress bar
        progress_bar = tqdm(
            total=len(urls),
            desc="Starting downloads...",
            unit="URL"
        )
        
        # Process URLs in parallel with progress bar
        start_time = time.time()
        
        # Prepare arguments for multiprocessing
        process_args = [(url, self.output_dir, i + 1, len(urls))
                       for i, url in enumerate(urls)]
        
        # Process with progress bar updates
        results = []
        failed_urls = []  # Track failed URLs for detailed reporting
        
        with Pool(processes=self.processes) as pool:
            for result in pool.imap(self.process_url, process_args):
                success, filename, status, url = result
                results.append((success, filename))
                
                base_filename = get_base_filename(filename)
                self.url_mapping[base_filename] = url
                
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
        
        # Save URL mapping to JSON file
        self.save_url_mapping()
        
        # Count results
        successful = sum(1 for success, _ in results if success)
        failed = len(results) - successful
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nDownload complete! Successful: {successful}, Failed: {failed}")
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
            
            # Retry failed URLs if requested
            if retry_failures:
                retry_result = self.retry_failed_urls(failed_urls)
                successful += retry_result["successful"]
                failed = retry_result["still_failed"]
        else:
            print(f"\n✅ All downloads completed successfully!")
        
        return {
            "successful": successful,
            "failed": failed,
            "failed_urls": failed_urls,
            "duration": duration
        }
    
    def save_url_mapping(self):
        """Save URL mapping to JSON file."""
        save_url_mapping(str(self.output_dir), self.url_mapping)
    
    def retry_failed_urls(self, failed_urls: List[Tuple[str, str, str]]) -> Dict[str, int]:
        """Retry downloading failed URLs, but skip certain types of permanent failures."""
        if not failed_urls:
            return {"successful": 0, "still_failed": 0}
        
        # Filter out failures that shouldn't be retried (permanent failures).
        # Note: We intentionally do NOT include "HTTP error 4" here because that
        # would also match retryable 429 (Too Many Requests) responses.
        permanent_failure_keywords = [
            "404", "Page not found",
            "HTTP error 410", "HTTP error 451",
            "Invalid or empty content",
        ]
        
        retryable_urls = []
        permanent_failures = []
        
        for filename, status, url in failed_urls:
            is_permanent = any(keyword in status for keyword in permanent_failure_keywords)
            if is_permanent:
                permanent_failures.append((filename, status, url))
            else:
                retryable_urls.append((filename, status, url))
        
        if permanent_failures:
            print(f"\nSkipping {len(permanent_failures)} permanent failures (404s, etc.)")
        
        if not retryable_urls:
            print("No retryable URLs found.")
            return {"successful": 0, "still_failed": len(permanent_failures)}
        
        print(f"\n=== RETRYING FAILED DOWNLOADS ===")
        print(f"Retrying {len(retryable_urls)} failed URLs (skipping {len(permanent_failures)} permanent failures)...")
        
        # Prepare arguments for retry
        retry_args = [(url, self.output_dir, i + 1, len(retryable_urls))
                     for i, (_, _, url) in enumerate(retryable_urls)]
        
        # Create progress bar for retry
        progress_bar = tqdm(total=len(retryable_urls), desc="Retrying...", unit="URL")
        
        successful_retries = 0
        still_failed = []
        
        with Pool(processes=self.processes) as pool:
            for result in pool.imap(self.process_url, retry_args):
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
        
        # Combine still failed with permanent failures
        all_failed = still_failed + permanent_failures
        
        print(f"\nRetry complete! Successfully retried: {successful_retries}, Still failed: {len(all_failed)}")
        print(f"  - Retryable failures: {len(still_failed)}")  
        print(f"  - Permanent failures (404s, etc.): {len(permanent_failures)}")
        
        if all_failed:
            print(f"\n=== STILL FAILED AFTER RETRY ===")
            for i, (filename, status, url) in enumerate(all_failed, 1):
                print(f"{i:2d}. {filename}")
                print(f"    URL: {url}")
                print(f"    Error: {status}")
                print()
        
        return {"successful": successful_retries, "still_failed": len(all_failed)}


class PDFDownloader(BaseDownloader):
    """Download web pages as PDFs using wkhtmltopdf."""
    
    def get_file_extension(self) -> str:
        return ".pdf"
    
    
    def download_single_url(self, url: str, output_path: Path) -> Tuple[bool, str]:
        """Download a single URL as PDF using wkhtmltopdf with try-fix-retry approach."""
        def attempt_pdf_download(target_url: str) -> Tuple[bool, str, bool]:
            """
            Attempt to download a URL as PDF.
            Returns (success, error_message, is_404_like_error)
            """
            command = (
                f'wkhtmltopdf --page-size A4 --margin-top 0.75in --margin-right 0.75in '
                f'--margin-bottom 0.75in --margin-left 0.75in --encoding UTF-8 '
                f'--load-error-handling ignore --load-media-error-handling ignore '
                f'--javascript-delay 2000 "{target_url}" "{output_path}"'
            )
            
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if result.returncode == 0:
                    return True, "Success", False
                else:
                    error_msg = f"wkhtmltopdf failed (return code: {result.returncode})"
                    if result.stderr:
                        stderr_text = result.stderr[:200]
                        error_msg += f" - {stderr_text}"
                        # Check for 404-like errors in stderr
                        is_404 = any(phrase in stderr_text.lower() for phrase in 
                                   ['404', 'not found', 'page not found', 'http error'])
                        return False, error_msg, is_404
                    return False, error_msg, False
                    
            except subprocess.TimeoutExpired:
                return False, "Timeout (120s)", False
        
        # Try original URL first
        success, error_msg, is_404 = attempt_pdf_download(url)
        
        if success:
            return True, "Success"
        
        # If it was a 404-like error, try to fix the URL and retry
        if is_404:
            fixed_url = fix_malformed_url(url)
            if fixed_url != url:
                print(f"Retrying PDF with fixed URL: {url} -> {fixed_url}")
                success, retry_error_msg, _ = attempt_pdf_download(fixed_url)
                if success:
                    return True, f"Success (fixed URL)"
                else:
                    return False, f"Original: {error_msg}; Fixed attempt: {retry_error_msg}"
        
        # Return original error if no fix was attempted or fix failed
        return False, error_msg


class HTMLDownloader(BaseDownloader):
    """Download web pages as HTML files using requests."""
    
    # Per-process retry policy for transient errors (429, 5xx)
    MAX_RETRIES = 5
    BASE_BACKOFF = 2.0  # seconds; exponential: BASE_BACKOFF * 2**attempt
    MAX_BACKOFF = 60.0  # cap on a single sleep

    def __init__(self, output_dir: str, processes: int = 4, delay: float = 1.0, timeout: int = 30):
        # HTML downloading can use parallel processes with rate limiting
        super().__init__(output_dir, processes=processes)
        self.delay = delay
        self.timeout = timeout
        
        if requests is None:
            raise ImportError("requests package is required for HTML downloads. Install with: pip install requests")
        
        self.session = requests.Session()
        
        # Wikipedia's User-Agent policy asks for a descriptive UA that identifies
        # the tool/operator and includes a contact URL or email. Generic browser
        # UAs are aggressively rate-limited. Operators may override via the
        # WIKIPEDIA_DOWNLOADER_UA env var.
        # See https://meta.wikimedia.org/wiki/User-Agent_policy
        default_ua = (
            "mvrag-inference-e2e/1.0 "
            "(https://github.com/; contact: rag-bench@local) "
            "python-requests"
        )
        self.session.headers.update({
            'User-Agent': os.environ.get('WIKIPEDIA_DOWNLOADER_UA', default_ua),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        })
    
    def get_file_extension(self) -> str:
        return ".html"
    
    
    def _compute_backoff(self, attempt: int, retry_after_header: Optional[str]) -> float:
        """Compute backoff delay honoring Retry-After if present, with jitter."""
        if retry_after_header:
            try:
                # Retry-After is usually an integer number of seconds
                return min(float(retry_after_header), self.MAX_BACKOFF)
            except (TypeError, ValueError):
                pass
        # Exponential backoff with full jitter
        cap = min(self.BASE_BACKOFF * (2 ** attempt), self.MAX_BACKOFF)
        return random.uniform(self.BASE_BACKOFF, cap)

    def download_single_url(self, url: str, output_path: Path) -> Tuple[bool, str]:
        """Download a single URL as HTML using requests with try-fix-retry approach."""
        def attempt_download(target_url: str) -> Tuple[bool, str, bool]:
            """
            Attempt to download a URL, retrying transient errors (429, 5xx).
            Returns (success, error_message, is_404_like_error)
            """
            last_error = "Unknown error"
            for attempt in range(self.MAX_RETRIES):
                try:
                    response = self.session.get(target_url, timeout=self.timeout)

                    # Retry on 429 (rate limit) and 5xx (server errors)
                    if response.status_code == 429 or 500 <= response.status_code < 600:
                        last_error = (
                            f"HTTP error {response.status_code}: "
                            f"{response.reason or 'transient error'}"
                        )
                        if attempt < self.MAX_RETRIES - 1:
                            sleep_for = self._compute_backoff(
                                attempt, response.headers.get('Retry-After')
                            )
                            print(
                                f"⏳ {response.status_code} for {target_url} "
                                f"(attempt {attempt + 1}/{self.MAX_RETRIES}); "
                                f"sleeping {sleep_for:.1f}s before retry"
                            )
                            time.sleep(sleep_for)
                            continue
                        return False, last_error, False

                    response.raise_for_status()

                    # Check if we got redirected to a different article
                    if response.url != target_url:
                        print(f"Redirected from {target_url} to {response.url}")

                    if 'charset' in response.headers.get('content-type', ''):
                        encoding = response.encoding
                    else:
                        encoding = 'utf-8'

                    html_content = response.content.decode(encoding, errors='ignore')

                    if 'wikipedia' not in html_content.lower() or len(html_content) < 1000:
                        return False, f"Invalid or empty content (length: {len(html_content)})", False

                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(html_content)

                    # Add rate limiting after successful download
                    time.sleep(self.delay)

                    return True, "Success", False

                except requests.HTTPError as e:
                    status = e.response.status_code if e.response is not None else None
                    if status == 404:
                        return False, f"Page not found (404): {target_url}", True
                    last_error = f"HTTP error {status}: {str(e)[:100]}"
                    return False, last_error, False
                except requests.RequestException as e:
                    last_error = f"Request error: {str(e)[:100]}"
                    if attempt < self.MAX_RETRIES - 1:
                        sleep_for = self._compute_backoff(attempt, None)
                        print(
                            f"⏳ Network error for {target_url} "
                            f"(attempt {attempt + 1}/{self.MAX_RETRIES}); "
                            f"sleeping {sleep_for:.1f}s before retry: {last_error}"
                        )
                        time.sleep(sleep_for)
                        continue
                    return False, last_error, False
                except Exception as e:
                    return False, f"Exception: {str(e)[:100]}", False

            return False, last_error, False
        
        # Try original URL first
        success, error_msg, is_404 = attempt_download(url)
        
        if success:
            return True, "Success"
        
        # If it was a 404-like error, try to fix the URL and retry
        if is_404:
            fixed_url = fix_malformed_url(url)
            if fixed_url != url:
                print(f"Retrying with fixed URL: {url} -> {fixed_url}")
                success, retry_error_msg, _ = attempt_download(fixed_url)
                if success:
                    return True, f"Success (fixed URL)"
                else:
                    return False, f"Original: {error_msg}; Fixed attempt: {retry_error_msg}"
        
        # Return original error if no fix was attempted or fix failed
        return False, error_msg
    
    def download_urls(self, urls: List[str], retry_failures: bool = True):
        """Download URLs with parallel processing and rate limiting."""
        return super().download_urls(urls, retry_failures)








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


_WIKI_URL_PATTERN = re.compile(
    # Match Wikipedia URLs, allowing commas that are part of the URL (e.g. in
    # disambiguation suffixes like "Quincy,_Massachusetts") while still treating
    # ", " / ",]" as list separators in the FRAMES dataset.
    r'https://en\.wikipedia\.org/wiki/(?:[^\s\],]|,(?=[^\s\],]))+'
)


def _strip_url_trailing_garbage(url: str) -> str:
    """Strip artifacts that are clearly not part of a Wikipedia URL.

    Wikipedia article titles legitimately end with characters that look like
    sentence punctuation (e.g. "_F.C.", "_Sr.", "Yahoo!", "Tick,_Tick..._Boom!",
    "Who's_Afraid_of_Virginia_Woolf%3F"), so we deliberately do NOT strip
    trailing ".", "!", "?", ":", ";", or quotes.

    The only artifact we strip is an unmatched trailing ")" that comes from
    Markdown link syntax like "[label](https://...)". Disambiguation URLs such
    as "/wiki/Quincy_(film)" keep their balanced parens intact.
    """
    while url.endswith(')') and url.count('(') < url.count(')'):
        url = url[:-1]
    return url


def extract_wikipedia_links(item):
    """Extract Wikipedia links from a FRAMES dataset item.

    The dataset stores lists of URLs as comma-separated strings (sometimes inside
    Markdown link syntax). Wikipedia URLs themselves can legitimately contain
    commas — most commonly in disambiguation suffixes like
    "Quincy,_Massachusetts" — so we cannot simply split on commas. Instead we
    match URLs greedily but treat a comma followed by whitespace/"]"/another
    comma as a list separator.
    """
    if isinstance(item, str):
        item = ast.literal_eval(item)

    links = []
    for entry in item:
        for match in _WIKI_URL_PATTERN.findall(entry):
            cleaned = _strip_url_trailing_garbage(match)
            if cleaned:
                links.append(cleaned)
    return links


def extract_urls_from_frames_dataset(tsv_path: str, max_urls: Optional[int] = None) -> List[str]:
    """Extract unique Wikipedia URLs from FRAMES dataset TSV file."""
    # Load dataset
    df = pd.read_csv(tsv_path, sep='\t')

    # Apply to all rows in df.wiki_links
    urls = set()
    for item in df.wiki_links:
        for link in extract_wikipedia_links(item):
            urls.add(link)

    urls = list(urls)
    
    # Limit number if specified
    if max_urls and max_urls > 0:
        urls = urls[:max_urls]
    
    return urls





def main():
    parser = argparse.ArgumentParser(
        description='Download Wikipedia pages as PDFs or HTML files from FRAMES dataset or other sources.\nBy default, URLs are validated before downloading to avoid 404 errors.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # Format selection
    parser.add_argument(
        '--format', 
        choices=['pdf', 'html'], 
        default='pdf',
        help='Output format: pdf or html (default: pdf)'
    )
    
    # URL sources (mutually exclusive)
    url_group = parser.add_mutually_exclusive_group()
    url_group.add_argument(
        '--tsv-path',
        help='Input TSV file from FRAMES dataset'
    )

    url_group.add_argument(
        '--urls', 
        nargs='+', 
        help='List of URLs to download'
    )
    url_group.add_argument(
        '--url-file', 
        help='File containing URLs (one per line)'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        help='Output directory (default: doc_pdf for PDF, doc_html for HTML)'
    )
    parser.add_argument(
        '--data-dir',
        default='frames-benchmark-dataset',
        help='Directory for dataset files (default: frames-benchmark-dataset)'
    )
    
    # Processing options
    parser.add_argument(
        '--max-files',
        type=int,
        help='Maximum number of URLs to process (default: all)'
    )
    parser.add_argument(
        '--processes',
        type=int,
        default=10,
        help='Number of parallel processes (default: 10). '
             'Wikipedia rate-limits aggressive HTML scrapers; '
             'consider --processes 4 for HTML.'
    )
    
    # HTML-specific options
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Per-process delay between HTML downloads in seconds (default: 1.0). '
             'Effective request rate ~= processes / delay.'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Timeout for HTML requests in seconds (default: 30)'
    )
    
    # Dataset options
    parser.add_argument(
        '--download-dataset',
        action='store_true',
        help='Download FRAMES dataset if no TSV provided'
    )

    args = parser.parse_args()

    # Set default output directory based on format
    if args.output_dir is None:
        args.output_dir = f"doc_{args.format}"

    # Get URLs from various sources
    urls = None
    
    if args.url_file:
        try:
            with open(args.url_file, 'r', encoding='utf-8') as f:
                urls = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(urls)} URLs from {args.url_file}")
        except Exception as e:
            print(f"Error loading URLs from file: {e}")
            return
    
    elif args.urls:
        urls = args.urls
    
    elif args.tsv_path or args.download_dataset:
        # Handle FRAMES dataset
        tsv_path = args.tsv_path
        
        if tsv_path is None and args.download_dataset:
            print("=== DOWNLOADING FRAMES DATASET ===")
            tsv_path = download_frames_dataset(args.data_dir)
            if tsv_path is None:
                print("❌ Failed to download FRAMES dataset. Exiting.")
                return
        elif tsv_path is None:
            print("❌ No TSV path provided. Use --tsv-path or --download-dataset")
            return
        
        urls = extract_urls_from_frames_dataset(tsv_path, args.max_files)
        print(f"Extracted {len(urls)} URLs from FRAMES dataset: {tsv_path}")
    
    else:
        # Default: download dataset
        print("=== DOWNLOADING FRAMES DATASET ===")
        tsv_path = download_frames_dataset(args.data_dir)
        if tsv_path is None:
            print("❌ Failed to download FRAMES dataset. Exiting.")
            return
        
        urls = extract_urls_from_frames_dataset(tsv_path, args.max_files)
        print(f"Extracted {len(urls)} URLs from FRAMES dataset: {tsv_path}")
    
    if not urls:
        print("No URLs found to download")
        return

    # Initialize appropriate downloader based on format
    if args.format == 'pdf':
        # Set XDG_RUNTIME_DIR for wkhtmltopdf
        user = getpass.getuser()
        xdg_runtime_dir = f"/tmp/runtime-{user}"
        os.environ["XDG_RUNTIME_DIR"] = xdg_runtime_dir
        
        downloader = PDFDownloader(args.output_dir, args.processes)
    
    elif args.format == 'html':
        if requests is None:
            print("❌ HTML format requires 'requests' package. Install with: pip install requests")
            return
        
        downloader = HTMLDownloader(
            output_dir=args.output_dir,
            processes=args.processes,
            delay=args.delay,
            timeout=args.timeout
        )
    
    else:
        print(f"❌ Unsupported format: {args.format}")
        return

    # Download URLs
    print(f"\n=== DOWNLOADING {len(urls)} URLs AS {args.format.upper()} ===")
    result = downloader.download_urls(urls, retry_failures=True)


if __name__ == "__main__":
    main()
