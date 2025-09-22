#!/usr/bin/env python3
import fitz
import os
import argparse
import re
import json
from pathlib import Path
from tqdm import tqdm

def extract_text_from_pdf(pdf_path):
    """Extract text from a single PDF file."""
    try:
        doc = fitz.open(pdf_path)
        extracted_text = []

        for page in doc:
            text = page.get_text()
            extracted_text.append(text)

        doc.close()
        return "\n".join(extracted_text)
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None

def split_into_passages(text, max_length=512, overlap=50):
    """
    Split text into passages suitable for ColBERT.

    Args:
        text: Input text to split
        max_length: Maximum length of each passage in characters
        overlap: Number of characters to overlap between passages

    Returns:
        List of passage texts
    """
    # Clean up the text
    text = re.sub(r'\s+', ' ', text.strip())

    if len(text) <= max_length:
        return [text] if text else []

    passages = []
    start = 0

    while start < len(text):
        end = start + max_length

        # If we're not at the end of the text, try to break at a sentence boundary
        if end < len(text):
            # Look for sentence endings within the last 100 characters
            search_start = max(start, end - 100)
            sentence_endings = ['.', '!', '?', '\n']

            best_break = end
            for i in range(end - 1, search_start - 1, -1):
                if text[i] in sentence_endings:
                    # Check if it's followed by whitespace and uppercase letter
                    if i + 1 < len(text) and text[i + 1].isspace():
                        # Look for the next non-whitespace character
                        j = i + 1
                        while j < len(text) and text[j].isspace():
                            j += 1
                        if j < len(text) and text[j].isupper():
                            best_break = i + 1
                            break

            end = best_break

        passage = text[start:end].strip()
        if passage:
            passages.append(passage)

        # Move start position with overlap
        start = end - overlap
        if start >= len(text):
            break

    return passages

def process_pdfs(input_dir, output_dir, json_file=None, max_passage_length=512, passage_overlap=50, max_files=None):
    """Process all PDFs in input directory and save text dumps to output directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all PDF files
    pdf_files = list(input_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
    
    # Limit number of files if max_files is specified
    if max_files is not None and max_files > 0:
        pdf_files = pdf_files[:max_files]
        print(f"Limited to first {len(pdf_files)} PDF files")

    print(f"Found {len(pdf_files)} PDF files to process")
    print(f"Input directory: {input_path}")
    print(f"Output directory: {output_path}")

    # Determine if JSON creation is enabled
    create_json = json_file is not None
    if create_json:
        print(f"JSON file: {json_file}")

    successful = 0
    failed = 0
    total_passages = 0

    # Initialize list to store passages data
    passages_data = []

    try:
        # Process each PDF file
        pbar = tqdm(pdf_files, desc="Processing PDFs")
        for pdf_file in pbar:
            # Create output filename (replace .pdf with .txt)
            output_file = output_path / (pdf_file.stem + ".txt")

            # Skip if output file already exists
            if output_file.exists():
                pbar.set_postfix({"Status": f"Skipped {pdf_file.name[:30]}..."})
                continue

            # Extract text from PDF
            text = extract_text_from_pdf(pdf_file)

            if text is not None:
                # Save text to file
                try:
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(text)
                    successful += 1

                    # Create passages data if requested
                    passage_count = 0
                    if create_json and text.strip():
                        passages = split_into_passages(text, max_passage_length, passage_overlap)
                        for i, passage in enumerate(passages):
                            clean_passage = re.sub(r'\s+', ' ', passage.strip())
                            if clean_passage:
                                # Store: index, pdf_filename, passage_text
                                passages_data.append({
                                    'index': total_passages + passage_count,
                                    'pdf_filename': pdf_file.name,
                                    'passage': clean_passage
                                })
                                passage_count += 1

                        total_passages += passage_count
                    
                    # Update progress bar with status
                    if create_json and passage_count > 0:
                        status = f"{total_passages} passages created (+ {passage_count})"
                        pbar.set_postfix({"Status": status})

                except Exception as e:
                    pbar.set_postfix({"Status": f"Failed {pdf_file.name[:30]}... ({e})"})
                    failed += 1
            else:
                pbar.set_postfix({"Status": f"Failed {pdf_file.name[:30]}... (no text)"})
                failed += 1
            
            total_passages += passage_count

    finally:
        # Save JSON file if creating one
        if create_json and passages_data:
            json_path = Path(json_file)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(passages_data, f, indent=2, ensure_ascii=False)

    print(f"\nProcessing complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    if create_json:
        print(f"Total passages created: {total_passages}")
        print(f"JSON file saved to: {json_file}")

def main():
    parser = argparse.ArgumentParser(description='Extract text from all PDFs in a directory')
    parser.add_argument('input_dir', help='Input directory containing PDF files')
    parser.add_argument('output_dir', help='Output directory for text files')
    parser.add_argument('--json-file', help='Output JSON file path for passages data (enables JSON creation)')
    parser.add_argument('--max-files', type=int, help='Maximum number of PDF files to process (default: all files)')
    parser.add_argument('--max-length', type=int, default=512, 
                       help='Maximum length of each passage in characters (default: 512)')
    parser.add_argument('--overlap', type=int, default=50,
                       help='Overlap between passages in characters (default: 50)')

    args = parser.parse_args()

    process_pdfs(args.input_dir, args.output_dir, 
                args.json_file, args.max_length, args.overlap, args.max_files)


if __name__ == "__main__":
    main()