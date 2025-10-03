#!/usr/bin/env python3
"""
Unified Document Text Extraction Pipeline

Extract text from PDF and HTML files using appropriate extractors based on file extensions.
Automatically detects file types and can process mixed directories.
Uses shared text splitting functionality for consistency.
"""

import fitz
import os
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
from abc import ABC, abstractmethod

try:
    from bs4 import BeautifulSoup, NavigableString, Tag
except ImportError as e:
    print(f"Warning: BeautifulSoup not available. HTML processing will be disabled.")
    print(f"Install with: pip install beautifulsoup4")
    BeautifulSoup = None

from text_splitter import split_into_passages, split_into_fixed_passages, create_passage_metadata
from utils import load_url_mapping, get_base_filename


class BaseDocumentExtractor(ABC):
    """Base class for document text extractors."""
    
    @abstractmethod
    def extract_text(self, file_path: str) -> Optional[str]:
        """Extract text from a document file."""
        pass
    
    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        pass


class PDFExtractor(BaseDocumentExtractor):
    """Extract text from PDF files using PyMuPDF (fitz)."""
    
    def extract_text(self, file_path: str) -> Optional[str]:
        """Extract text from a single PDF file."""
        try:
            doc = fitz.open(file_path)
            extracted_text = []

            for page in doc:
                text = page.get_text()
                extracted_text.append(text)

            doc.close()
            return "\n".join(extracted_text)
            
        except Exception as e:
            print(f"Error processing PDF {file_path}: {e}")
            return None
    
    def get_supported_extensions(self) -> List[str]:
        return ['.pdf']


class HTMLExtractor(BaseDocumentExtractor):
    """Extract text from HTML files using BeautifulSoup."""
    
    def __init__(self, preserve_tables: bool = True, preserve_lists: bool = True):
        self.preserve_tables = preserve_tables
        self.preserve_lists = preserve_lists
        
        if BeautifulSoup is None:
            raise ImportError("BeautifulSoup is required for HTML processing. Install with: pip install beautifulsoup4")
    
    def extract_text(self, file_path: str) -> Optional[str]:
        """Extract text from a single HTML file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            return self.extract_text_from_html(html_content)
            
        except Exception as e:
            print(f"Error processing HTML {file_path}: {e}")
            return None
    
    def extract_text_from_html(self, html_content: str) -> str:
        """Extract text from HTML content with semantic preservation."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()
        
        # Process the content
        text_parts = []
        
        # Extract main content (try common content containers first)
        main_content = (soup.find('div', {'id': 'mw-content-text'}) or  # Wikipedia
                       soup.find('main') or 
                       soup.find('article') or
                       soup.find('div', {'class': 'content'}) or
                       soup.body or soup)
        
        if main_content:
            text_parts.extend(self._extract_from_element(main_content))
        
        # Join and clean up the text
        full_text = ' '.join(text_parts)
        return self._clean_text(full_text)
    
    def _extract_from_element(self, element) -> List[str]:
        """Extract text from an HTML element, preserving structure."""
        text_parts = []
        
        if isinstance(element, NavigableString):
            text = str(element).strip()
            if text:
                text_parts.append(text)
        elif isinstance(element, Tag):
            # Handle different tag types
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                # Headers get extra spacing
                text = element.get_text().strip()
                if text:
                    text_parts.append(f"\n{text}\n")
            elif element.name == 'p':
                # Paragraphs
                text = element.get_text().strip()
                if text:
                    text_parts.append(f"{text}\n")
            elif element.name == 'table' and self.preserve_tables:
                # Tables - extract as structured text
                table_text = self._extract_table_text(element)
                if table_text:
                    text_parts.append(f"\n{table_text}\n")
            elif element.name in ['ul', 'ol'] and self.preserve_lists:
                # Lists
                list_text = self._extract_list_text(element)
                if list_text:
                    text_parts.append(f"\n{list_text}\n")
            elif element.name in ['br']:
                text_parts.append('\n')
            elif element.name in ['div', 'span', 'section', 'article']:
                # Container elements - process children
                for child in element.children:
                    text_parts.extend(self._extract_from_element(child))
            else:
                # For other elements, just get the text
                text = element.get_text().strip()
                if text:
                    text_parts.append(text)
        
        return text_parts
    
    def _extract_table_text(self, table) -> str:
        """Extract text from a table element."""
        rows = []
        for tr in table.find_all('tr'):
            cells = []
            for cell in tr.find_all(['td', 'th']):
                cell_text = cell.get_text().strip()
                cells.append(cell_text)
            if cells:
                rows.append(' | '.join(cells))
        return '\n'.join(rows)
    
    def _extract_list_text(self, list_elem) -> str:
        """Extract text from a list element."""
        items = []
        for li in list_elem.find_all('li', recursive=False):
            item_text = li.get_text().strip()
            if item_text:
                prefix = '- ' if list_elem.name == 'ul' else f"{len(items) + 1}. "
                items.append(f"{prefix}{item_text}")
        return '\n'.join(items)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        # Replace multiple whitespaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def get_supported_extensions(self) -> List[str]:
        return ['.html', '.htm']


class DocumentProcessor:
    """Unified document processor that handles both PDF and HTML files."""
    
    def __init__(self, preserve_tables: bool = True, preserve_lists: bool = True):
        self.extractors = {
            '.pdf': PDFExtractor(),
        }
        
        # Only add HTML extractor if BeautifulSoup is available
        if BeautifulSoup is not None:
            self.extractors.update({
                '.html': HTMLExtractor(preserve_tables, preserve_lists),
                '.htm': HTMLExtractor(preserve_tables, preserve_lists),
            })
        
        self.url_mapping = {}
    
    def get_supported_extensions(self) -> List[str]:
        """Get all supported file extensions."""
        extensions = []
        for extractor in self.extractors.values():
            extensions.extend(extractor.get_supported_extensions())
        return list(set(extensions))
    
    def process_documents(self, input_dir: str, output_dir: str, json_file: Optional[str] = None,
                         max_passage_length: int = 512, passage_overlap: int = 50, 
                         fixed_length: Optional[int] = None, fixed_overlap: Optional[int] = None,
                         max_files: Optional[int] = None):
        """
        Process documents in a directory, extracting text and splitting into passages.
        
        Args:
            input_dir: Directory containing document files
            output_dir: Directory to save extracted text files
            json_file: Optional JSON file to save passage metadata
            max_passage_length: Maximum passage length for variable-length splitting
            passage_overlap: Overlap between passages for variable-length splitting
            fixed_length: Use fixed-length passages instead of variable-length
            fixed_overlap: Overlap for fixed-length passages
            max_files: Maximum number of files to process (for testing)
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.url_mapping = load_url_mapping(input_dir)

        # Find all supported document files
        supported_extensions = self.get_supported_extensions()
        document_files = []
        
        for ext in supported_extensions:
            pattern = f"*{ext}"
            document_files.extend(input_path.glob(pattern))
        
        # Sort for consistent processing order
        document_files = sorted(document_files)
        
        if not document_files:
            return
        
        if max_files:
            document_files = document_files[:max_files]
        
        all_passages = []
        passage_id = 0

        for doc_file in tqdm(document_files, desc="Processing documents"):
            file_extension = doc_file.suffix.lower()
            
            if file_extension not in self.extractors:
                continue
            
            extractor = self.extractors[file_extension]
            text = extractor.extract_text(str(doc_file))
            
            if text is None:
                continue
            
            # Determine output filename (change extension to .txt)
            output_filename = doc_file.stem + ".txt"
            output_file_path = output_path / output_filename
            
            # Split into passages
            if fixed_length:
                passages = split_into_fixed_passages(text, fixed_length, fixed_overlap or 32)
            else:
                passages = split_into_passages(text, max_passage_length, passage_overlap)
            
            # Save text file
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Add passages to collection if JSON output requested
            if json_file:
                base_filename = get_base_filename(doc_file.name)
                original_url = self.url_mapping.get(base_filename, "")
                
                for i, passage in enumerate(passages):
                    passage_metadata = create_passage_metadata(
                        doc_file.name, 
                        passage_id,
                        original_url=original_url
                    )
                    
                    all_passages.append({
                        **passage_metadata,
                        'passage': passage,
                    })
                    passage_id += 1

        # Save JSON file if requested
        if json_file and all_passages:
            json_path = Path(json_file)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(all_passages, f, indent=2, ensure_ascii=False)
            
            print(f"Saved {len(all_passages)} passages to {json_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from PDF and HTML files and split into passages",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument("input_dir", help="Directory containing document files (PDF/HTML)")
    parser.add_argument("output_dir", help="Directory to save text files")
    parser.add_argument("--json", help="JSON file to save passage data")
    parser.add_argument("--max-length", type=int, default=512, 
                       help="Maximum passage length in characters (default: 512)")
    parser.add_argument("--overlap", type=int, default=50,
                       help="Overlap between passages in characters (default: 50)")
    parser.add_argument("--fixed-length", type=int,
                       help="Use fixed-length passages instead of variable-length")
    parser.add_argument("--fixed-overlap", type=int, default=32,
                       help="Overlap for fixed-length passages (default: 32)")
    parser.add_argument("--max-files", type=int,
                       help="Maximum number of files to process (for testing)")
    parser.add_argument("--no-tables", action="store_true",
                       help="Don't preserve table structure (HTML only)")
    parser.add_argument("--no-lists", action="store_true", 
                       help="Don't preserve list structure (HTML only)")

    args = parser.parse_args()

    processor = DocumentProcessor(
        preserve_tables=not args.no_tables,
        preserve_lists=not args.no_lists
    )
    
    processor.process_documents(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        json_file=args.json,
        max_passage_length=args.max_length,
        passage_overlap=args.overlap,
        fixed_length=args.fixed_length,
        fixed_overlap=args.fixed_overlap,
        max_files=args.max_files
    )


if __name__ == "__main__":
    main()