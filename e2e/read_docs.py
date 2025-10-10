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
    """Extract text from HTML files using BeautifulSoup with focus on retrieval quality."""
    
    def __init__(self, preserve_tables: bool = True, preserve_lists: bool = True, 
                 text_boundary: str = "sentence"):
        """
        Initialize HTML extractor with configurable options.
        
        Args:
            preserve_tables: Whether to preserve table structure
            preserve_lists: Whether to preserve list structure  
            text_boundary: Text boundary optimization - "sentence" (default), "word", or "none"
        """
        if BeautifulSoup is None:
            raise ImportError("BeautifulSoup is required for HTML processing. Install with: pip install beautifulsoup4")
        
        self.preserve_tables = preserve_tables
        self.preserve_lists = preserve_lists
        self.text_boundary = text_boundary
        
        if text_boundary not in ["sentence", "word", "none"]:
            raise ValueError("text_boundary must be 'sentence', 'word', or 'none'")
    
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
        """Extract clean text optimized for retrieval systems."""
        # Use lxml parser for speed (fallback to html.parser if not available)
        try:
            soup = BeautifulSoup(html_content, 'lxml')
        except:
            soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove noise elements completely
        for element in soup(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()
        
        # Remove Wikipedia-specific metadata and navigation
        self._remove_wikipedia_metadata(soup)
        
        # Extract main content using priority order
        main_content = self._find_main_content(soup)
        
        # Get plain text with sentence separation
        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
        else:
            text = soup.get_text(separator=' ', strip=True)
        
        # Clean and normalize the text
        return self._clean_text(text)
    
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
        """Clean and normalize extracted text with configurable boundary optimization."""
        # Basic normalization
        import unicodedata
        text = unicodedata.normalize('NFKC', text)
        
        # Replace various whitespace characters with standard space
        text = re.sub(r'[\u00A0\u2000-\u200B\u2028\u2029]', ' ', text)
        
        # Clean up whitespace
        text = text.strip().replace('\r', '\n')
        text = re.sub(r' +', ' ', text)  # Multiple spaces -> single space
        text = re.sub(r'\n+', '\n', text)  # Multiple newlines -> single newline
        
        # Remove empty lines and extra spacing
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        # Apply boundary optimization based on setting
        if self.text_boundary == "sentence":
            text = self._optimize_sentence_boundaries(text)
        elif self.text_boundary == "word":
            text = self._optimize_word_boundaries(text)
        # "none" - no boundary optimization
        
        return text
    
    def _optimize_sentence_boundaries(self, text: str) -> str:
        """Optimize text for sentence-level splitting and retrieval."""
        # Add space after sentence endings if missing
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        
        # Handle common abbreviations that shouldn't split sentences
        # (e.g., "Mr.", "Dr.", "etc.", "U.S.")
        abbrev_pattern = r'\b(Mr|Mrs|Dr|Prof|etc|vs|Inc|Ltd|Corp|U\.S|U\.K|E\.g|I\.e)\.(\s+)([a-z])'
        text = re.sub(abbrev_pattern, r'\1.\2\3', text, flags=re.IGNORECASE)
        
        return text
    
    def _optimize_word_boundaries(self, text: str) -> str:
        """Optimize text for word-level processing and retrieval."""
        # Ensure proper spacing around punctuation for better tokenization
        text = re.sub(r'([.!?,:;])([A-Za-z])', r'\1 \2', text)
        
        # Handle hyphenated words - keep them as single tokens
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1-\2', text)
        
        # Normalize quotation marks and other punctuation
        text = text.replace('"', '"').replace('"', '"')  # Smart quotes to regular quotes
        text = text.replace(''', "'").replace(''', "'")  # Smart apostrophes to regular apostrophes
        
        # Ensure consistent spacing
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def _remove_wikipedia_metadata(self, soup):
        """Remove Wikipedia-specific metadata and navigation elements."""
        # Wikipedia-specific noise removal
        selectors_to_remove = [
            # Navigation and interface elements
            '#mw-navigation', '.navbox', '.navigation-box', 
            '.sidebar', '.infobox', '.ambox', '.tmbox',
            # Edit links and metadata
            '.mw-editsection', '.edit-section', '.editlink',
            # References and citations (keep text but remove citation numbers)
            'sup.reference', '.reference', '.citation',
            # Disambiguation and hatnotes  
            '.hatnote', '.dablink', '.rellink',
            # Categories and external links boxes
            '#catlinks', '.catlinks', '.external-links',
            # Table of contents (often not needed for retrieval)
            '#toc', '.toc',
            # Image captions and metadata (keep main text)
            '.thumbcaption .metadata', '.image-metadata'
        ]
        
        for selector in selectors_to_remove:
            for element in soup.select(selector):
                element.decompose()
    
    def _find_main_content(self, soup):
        """Find the main content area with fallback strategy."""
        # Priority order for content detection
        content_selectors = [
            '#mw-content-text .mw-parser-output',  # Wikipedia main content
            '#mw-content-text',                    # Wikipedia content wrapper
            'main',                                # HTML5 main element
            'article',                             # HTML5 article element
            '.content',                            # Generic content class
            '#content',                            # Generic content ID
            'body'                                 # Last resort
        ]
        
        for selector in content_selectors:
            content = soup.select_one(selector)
            if content:
                return content
        
        # Final fallback
        return soup
    
    def get_supported_extensions(self) -> List[str]:
        return ['.html', '.htm']


class DocumentProcessor:
    """Unified document processor that handles both PDF and HTML files."""
    
    def __init__(self, preserve_tables: bool = True, preserve_lists: bool = True, 
                 text_boundary: str = "sentence", benchmark: bool = False):
        """
        Initialize document processor.
        
        Args:
            preserve_tables: Whether to preserve table structure (HTML only)
            preserve_lists: Whether to preserve list structure (HTML only) 
            text_boundary: Text boundary optimization - "sentence" (default), "word", or "none"
            benchmark: Enable performance monitoring
        """
        self.extractors = {
            '.pdf': PDFExtractor(),
        }
        
        # Only add HTML extractor if BeautifulSoup is available
        if BeautifulSoup is not None:
            self.extractors.update({
                '.html': HTMLExtractor(preserve_tables, preserve_lists, text_boundary),
                '.htm': HTMLExtractor(preserve_tables, preserve_lists, text_boundary),
            })
        
        self.url_mapping = {}
        self.benchmark = benchmark
        self.monitor = None
        
        # Initialize monitoring if benchmark mode enabled
        if self.benchmark:
            from ingestion_monitor import IngestionMonitor
            self.monitor = IngestionMonitor()
    
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

        # Initialize monitoring if enabled
        if self.benchmark and self.monitor:
            self.monitor.start_ingestion()

        for doc_file in tqdm(document_files, desc="Processing documents"):
            file_extension = doc_file.suffix.lower()
            
            if file_extension not in self.extractors:
                continue
            
            # Process single document with optional monitoring
            text = self._process_document(doc_file, file_extension)
            if text is None:
                continue
            
            # Split text into passages with optional monitoring
            passages = self._process_text_chunking(text, fixed_length, fixed_overlap, 
                                                 max_passage_length, passage_overlap)
            
            # Save text file
            output_filename = doc_file.stem + ".txt"
            output_file_path = output_path / output_filename
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Add passages to collection if JSON output requested
            if json_file:
                passage_id = self._add_passages_to_collection(
                    doc_file, passages, all_passages, passage_id)

        # Finalize monitoring and report
        self._report_processing_performance()
        
        # Save JSON file if requested
        if json_file and all_passages:
            json_path = Path(json_file)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(all_passages, f, indent=2, ensure_ascii=False)
            
            print(f"Saved {len(all_passages)} passages to {json_file}")
    
    def _process_document(self, doc_file: Path, file_extension: str) -> Optional[str]:
        """Process a single document with optional monitoring."""
        extractor = self.extractors[file_extension]
        
        if self.benchmark and self.monitor:
            component_name = "html_parsing" if file_extension in ['.html', '.htm'] else "pdf_parsing"
            file_size = doc_file.stat().st_size
            with self.monitor.track_component(component_name, input_size_bytes=file_size, 
                                             items_count=1, is_pipeline_input=True):
                return extractor.extract_text(str(doc_file))
        else:
            return extractor.extract_text(str(doc_file))
    
    def _process_text_chunking(self, text: str, fixed_length: Optional[int], fixed_overlap: Optional[int],
                              max_passage_length: int, passage_overlap: int) -> List[str]:
        """Process text chunking with optional monitoring."""
        def chunk_func():
            if fixed_length:
                return split_into_fixed_passages(text, fixed_length, fixed_overlap or 32)
            else:
                return split_into_passages(text, max_passage_length, passage_overlap)
        
        if self.benchmark and self.monitor:
            text_size = len(text.encode('utf-8'))
            with self.monitor.track_component("text_chunking", input_size_bytes=text_size, 
                                             items_count=1, text_only=True) as ctx:
                passages = chunk_func()
                ctx.add_text_bytes(text_size)
                return passages
        else:
            return chunk_func()
    
    def _add_passages_to_collection(self, doc_file: Path, passages: List[str], 
                                   all_passages: List[Dict], passage_id: int) -> int:
        """Add passages to collection and return updated passage_id."""
        base_filename = get_base_filename(doc_file.name)
        original_url = self.url_mapping.get(base_filename, "")
        
        for passage in passages:
            passage_metadata = create_passage_metadata(
                doc_file.name, passage_id, original_url=original_url)
            
            all_passages.append({
                **passage_metadata,
                'passage': passage,
            })
            passage_id += 1
        
        return passage_id
    
    def _report_processing_performance(self):
        """Report processing performance if monitoring is enabled."""
        if self.benchmark and self.monitor:
            print("\n=== Document Processing Performance ===")
            self.monitor.print_summary()


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
    parser.add_argument("--text-boundary", choices=["sentence", "word", "none"], 
                       default="sentence",
                       help="Text boundary optimization: 'sentence' (default), 'word', or 'none'")
    parser.add_argument("--benchmark", action="store_true",
                       help="Enable performance monitoring and detailed component analysis")

    args = parser.parse_args()

    processor = DocumentProcessor(
        preserve_tables=not args.no_tables,
        preserve_lists=not args.no_lists,
        text_boundary=args.text_boundary,
        benchmark=args.benchmark
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
