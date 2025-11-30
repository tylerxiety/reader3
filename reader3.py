"""
Parses an EPUB file into a structured object that can be used to serve the book via a web interface.
"""

import os
import pickle
import shutil
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from urllib.parse import unquote

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, Comment
from mobi import extract as mobi_extract

# --- Data structures ---

@dataclass
class ChapterContent:
    """
    Represents a physical file in the EPUB (Spine Item).
    A single file might contain multiple logical chapters (TOC entries).
    """
    id: str           # Internal ID (e.g., 'item_1')
    href: str         # Filename (e.g., 'part01.html')
    title: str        # Best guess title from file
    content: str      # Cleaned HTML with rewritten image paths
    text: str         # Plain text for search/LLM context
    order: int        # Linear reading order


@dataclass
class TOCEntry:
    """Represents a logical entry in the navigation sidebar."""
    title: str
    href: str         # original href (e.g., 'part01.html#chapter1')
    file_href: str    # just the filename (e.g., 'part01.html')
    anchor: str       # just the anchor (e.g., 'chapter1'), empty if none
    children: List['TOCEntry'] = field(default_factory=list)


@dataclass
class BookMetadata:
    """Metadata"""
    title: str
    language: str
    authors: List[str] = field(default_factory=list)
    description: Optional[str] = None
    publisher: Optional[str] = None
    date: Optional[str] = None
    identifiers: List[str] = field(default_factory=list)
    subjects: List[str] = field(default_factory=list)


@dataclass
class Book:
    """The Master Object to be pickled."""
    metadata: BookMetadata
    spine: List[ChapterContent]  # The actual content (linear files)
    toc: List[TOCEntry]          # The navigation tree
    images: Dict[str, str]       # Map: original_path -> local_path

    # Meta info
    source_file: str
    processed_at: str
    version: str = "3.0"


# --- Utilities ---

def clean_html_content(soup: BeautifulSoup) -> BeautifulSoup:

    # Remove dangerous/useless tags
    for tag in soup(['script', 'style', 'iframe', 'video', 'nav', 'form', 'button']):
        tag.decompose()

    # Remove HTML comments
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    # Remove input tags
    for tag in soup.find_all('input'):
        tag.decompose()

    return soup


def extract_plain_text(soup: BeautifulSoup) -> str:
    """Extract clean text for LLM/Search usage."""
    text = soup.get_text(separator=' ')
    # Collapse whitespace
    return ' '.join(text.split())


def parse_toc_recursive(toc_list, depth=0) -> List[TOCEntry]:
    """
    Recursively parses the TOC structure from ebooklib.
    """
    result = []

    for item in toc_list:
        # ebooklib TOC items are either `Link` objects or tuples (Section, [Children])
        if isinstance(item, tuple):
            section, children = item
            entry = TOCEntry(
                title=section.title,
                href=section.href,
                file_href=section.href.split('#')[0],
                anchor=section.href.split('#')[1] if '#' in section.href else "",
                children=parse_toc_recursive(children, depth + 1)
            )
            result.append(entry)
        elif isinstance(item, epub.Link):
            entry = TOCEntry(
                title=item.title,
                href=item.href,
                file_href=item.href.split('#')[0],
                anchor=item.href.split('#')[1] if '#' in item.href else ""
            )
            result.append(entry)
        # Note: ebooklib sometimes returns direct Section objects without children
        elif isinstance(item, epub.Section):
             entry = TOCEntry(
                title=item.title,
                href=item.href,
                file_href=item.href.split('#')[0],
                anchor=item.href.split('#')[1] if '#' in item.href else ""
            )
             result.append(entry)

    return result


def get_fallback_toc(book_obj) -> List[TOCEntry]:
    """
    If TOC is missing, build a flat one from the Spine.
    """
    toc = []
    for item in book_obj.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            name = item.get_name()
            # Try to guess a title from the content or ID
            title = item.get_name().replace('.html', '').replace('.xhtml', '').replace('_', ' ').title()
            toc.append(TOCEntry(title=title, href=name, file_href=name, anchor=""))
    return toc


def extract_metadata_robust(book_obj) -> BookMetadata:
    """
    Extracts metadata handling both single and list values.
    """
    def get_list(key):
        data = book_obj.get_metadata('DC', key)
        return [x[0] for x in data] if data else []

    def get_one(key):
        data = book_obj.get_metadata('DC', key)
        return data[0][0] if data else None

    return BookMetadata(
        title=get_one('title') or "Untitled",
        language=get_one('language') or "en",
        authors=get_list('creator'),
        description=get_one('description'),
        publisher=get_one('publisher'),
        date=get_one('date'),
        identifiers=get_list('identifier'),
        subjects=get_list('subject')
    )


# --- Main Conversion Logic ---

def process_epub(epub_path: str, output_dir: str) -> Book:

    # 1. Load Book
    print(f"Loading {epub_path}...")
    book = epub.read_epub(epub_path)

    # 2. Extract Metadata
    metadata = extract_metadata_robust(book)

    # 3. Prepare Output Directories
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    # 4. Extract Images & Build Map
    print("Extracting images...")
    image_map = {} # Key: internal_path, Value: local_relative_path

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_IMAGE:
            # Normalize filename
            original_fname = os.path.basename(item.get_name())
            # Sanitize filename for OS
            safe_fname = "".join([c for c in original_fname if c.isalpha() or c.isdigit() or c in '._-']).strip()

            # Save to disk
            local_path = os.path.join(images_dir, safe_fname)
            with open(local_path, 'wb') as f:
                f.write(item.get_content())

            # Map keys: We try both the full internal path and just the basename
            # to be robust against messy HTML src attributes
            rel_path = f"images/{safe_fname}"
            image_map[item.get_name()] = rel_path
            image_map[original_fname] = rel_path

    # 5. Process TOC
    print("Parsing Table of Contents...")
    toc_structure = parse_toc_recursive(book.toc)
    if not toc_structure:
        print("Warning: Empty TOC, building fallback from Spine...")
        toc_structure = get_fallback_toc(book)

    # 6. Process Content (Spine-based to preserve HTML validity)
    print("Processing chapters...")
    spine_chapters = []

    # We iterate over the spine (linear reading order)
    for i, spine_item in enumerate(book.spine):
        item_id, linear = spine_item
        item = book.get_item_with_id(item_id)

        if not item:
            continue

        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            # Raw content
            raw_content = item.get_content().decode('utf-8', errors='ignore')
            soup = BeautifulSoup(raw_content, 'html.parser')

            # A. Fix Images
            for img in soup.find_all('img'):
                src = img.get('src', '')
                if not src: continue

                # Decode URL (part01/image%201.jpg -> part01/image 1.jpg)
                src_decoded = unquote(src)
                filename = os.path.basename(src_decoded)

                # Try to find in map
                if src_decoded in image_map:
                    img['src'] = image_map[src_decoded]
                elif filename in image_map:
                    img['src'] = image_map[filename]

            # B. Clean HTML
            soup = clean_html_content(soup)

            # C. Extract Body Content only
            body = soup.find('body')
            if body:
                # Extract inner HTML of body
                final_html = "".join([str(x) for x in body.contents])
            else:
                final_html = str(soup)

            # D. Create Object
            chapter = ChapterContent(
                id=item_id,
                href=item.get_name(), # Important: This links TOC to Content
                title=f"Section {i+1}", # Fallback, real titles come from TOC
                content=final_html,
                text=extract_plain_text(soup),
                order=i
            )
            spine_chapters.append(chapter)

    # 7. Final Assembly
    final_book = Book(
        metadata=metadata,
        spine=spine_chapters,
        toc=toc_structure,
        images=image_map,
        source_file=os.path.basename(epub_path),
        processed_at=datetime.now().isoformat()
    )

    return final_book


def save_to_pickle(book: Book, output_dir: str):
    p_path = os.path.join(output_dir, 'book.pkl')
    with open(p_path, 'wb') as f:
        pickle.dump(book, f)
    print(f"Saved structured data to {p_path}")


# --- MOBI Processing ---

def _locate_content_root(extract_dir: str) -> str:
    """
    Find the best content root directory. Prefer mobi8/OEBPS > mobi8 > mobi7 > root.
    """
    # Try mobi8/OEBPS first (best quality for KF8)
    mobi8_oebps = os.path.join(extract_dir, 'mobi8', 'OEBPS')
    if os.path.exists(mobi8_oebps):
        return mobi8_oebps

    # Try mobi8
    mobi8 = os.path.join(extract_dir, 'mobi8')
    if os.path.exists(mobi8):
        return mobi8

    # Try mobi7
    mobi7 = os.path.join(extract_dir, 'mobi7')
    if os.path.exists(mobi7):
        return mobi7

    # Fall back to root
    return extract_dir


def _find_opf_file(content_root: str) -> Optional[str]:
    """
    Locate content.opf file in the content root or subdirectories.
    """
    # Check root first
    opf_path = os.path.join(content_root, 'content.opf')
    if os.path.exists(opf_path):
        return opf_path

    # Search recursively
    for root, dirs, files in os.walk(content_root):
        for file in files:
            if file.endswith('.opf'):
                return os.path.join(root, file)

    return None


def _find_ncx_file(content_root: str) -> Optional[str]:
    """
    Locate toc.ncx file in the content root or subdirectories.
    """
    # Check root first
    ncx_path = os.path.join(content_root, 'toc.ncx')
    if os.path.exists(ncx_path):
        return ncx_path

    # Search recursively
    for root, dirs, files in os.walk(content_root):
        for file in files:
            if file.endswith('.ncx'):
                return os.path.join(root, file)

    return None


def _extract_mobi_metadata(opf_path: str) -> BookMetadata:
    """
    Extract metadata from OPF file using Dublin Core elements.
    """
    ns = {
        'opf': 'http://www.idpf.org/2007/opf',
        'dc': 'http://purl.org/dc/elements/1.1/'
    }

    tree = ET.parse(opf_path)
    root = tree.getroot()

    # Helper to get text from element
    def get_text(xpath, default=None):
        elem = root.find(xpath, ns)
        return elem.text if elem is not None else default

    # Helper to get all texts from elements
    def get_all_texts(xpath):
        return [elem.text for elem in root.findall(xpath, ns) if elem.text]

    return BookMetadata(
        title=get_text('.//dc:title', 'Untitled'),
        language=get_text('.//dc:language', 'en'),
        authors=get_all_texts('.//dc:creator'),
        description=get_text('.//dc:description'),
        publisher=get_text('.//dc:publisher'),
        date=get_text('.//dc:date'),
        identifiers=get_all_texts('.//dc:identifier'),
        subjects=get_all_texts('.//dc:subject')
    )


def _extract_spine_order(opf_path: str, content_root: str) -> List[tuple]:
    """
    Extract reading order from OPF spine.
    Returns list of (href, item_id) tuples.
    """
    ns = {
        'opf': 'http://www.idpf.org/2007/opf'
    }

    tree = ET.parse(opf_path)
    root = tree.getroot()

    # Build manifest map: id -> href
    manifest = {}
    for item in root.findall('.//opf:manifest/opf:item', ns):
        item_id = item.get('id')
        href = item.get('href')
        if item_id and href:
            manifest[item_id] = href

    # Extract spine order
    spine_items = []
    for itemref in root.findall('.//opf:spine/opf:itemref', ns):
        idref = itemref.get('idref')
        if idref and idref in manifest:
            href = manifest[idref]
            spine_items.append((href, idref))

    return spine_items


def _parse_navpoint_recursive(navpoint, ns: dict, content_root: str) -> TOCEntry:
    """
    Recursively parse a navPoint element from NCX.
    """
    # Get title
    text_elem = navpoint.find('.//ncx:text', ns)
    title = text_elem.text if text_elem is not None else "Untitled"

    # Get href
    content_elem = navpoint.find('.//ncx:content', ns)
    href = content_elem.get('src') if content_elem is not None else ""

    # Split href into file and anchor
    file_href = href.split('#')[0] if href else ""
    anchor = href.split('#')[1] if '#' in href else ""

    # Parse children recursively
    children = []
    for child_navpoint in navpoint.findall('ncx:navPoint', ns):
        children.append(_parse_navpoint_recursive(child_navpoint, ns, content_root))

    return TOCEntry(
        title=title,
        href=href,
        file_href=file_href,
        anchor=anchor,
        children=children
    )


def _parse_ncx_toc(ncx_path: str, content_root: str) -> List[TOCEntry]:
    """
    Parse NCX file to extract table of contents.
    """
    ns = {'ncx': 'http://www.daisy.org/z3986/2005/ncx/'}

    tree = ET.parse(ncx_path)
    root = tree.getroot()

    # Find all top-level navPoints
    toc_entries = []
    navmap = root.find('.//ncx:navMap', ns)
    if navmap is not None:
        for navpoint in navmap.findall('ncx:navPoint', ns):
            entry = _parse_navpoint_recursive(navpoint, ns, content_root)
            toc_entries.append(entry)

    return toc_entries


def _extract_mobi_images(content_root: str, output_dir: str) -> Dict[str, str]:
    """
    Extract images from MOBI content to output directory.
    Returns map of original_path -> local_relative_path.
    """
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    image_map = {}

    # Search for image directories
    for root_dir, dirs, files in os.walk(content_root):
        for file in files:
            # Check if it's an image file
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp')):
                original_path = os.path.join(root_dir, file)

                # Sanitize filename
                safe_fname = "".join([c for c in file if c.isalpha() or c.isdigit() or c in '._-']).strip()
                if not safe_fname:
                    safe_fname = f"image_{hash(file)}.jpg"

                # Copy to output
                local_path = os.path.join(images_dir, safe_fname)
                shutil.copy2(original_path, local_path)

                # Build map with multiple keys for robustness
                rel_path = f"images/{safe_fname}"
                image_map[file] = rel_path  # Just filename

                # Relative path from content root
                rel_from_root = os.path.relpath(original_path, content_root)
                image_map[rel_from_root] = rel_path

                # Also try normalized path
                image_map[rel_from_root.replace('\\', '/')] = rel_path

    return image_map


def _build_fallback_mobi_toc(spine_items: List[tuple]) -> List[TOCEntry]:
    """
    Build a flat TOC from spine items when NCX is missing.
    """
    toc = []
    for i, (href, item_id) in enumerate(spine_items):
        # Extract a title from the filename
        title = os.path.splitext(os.path.basename(href))[0]
        title = title.replace('_', ' ').replace('-', ' ').title()
        if not title:
            title = f"Section {i+1}"

        toc.append(TOCEntry(
            title=title,
            href=href,
            file_href=href,
            anchor="",
            children=[]
        ))

    return toc


def process_mobi(mobi_path: str, output_dir: str) -> Book:
    """
    Process a MOBI file into a Book object (mirrors process_epub).
    """
    print(f"Loading {mobi_path}...")

    # Extract MOBI to temporary directory
    temp_dir = None
    try:
        temp_dir, epub_path = mobi_extract(mobi_path)
        print(f"Extracted MOBI to temporary directory")

        # Locate content root (prefer mobi8/OEBPS > mobi8 > mobi7)
        content_root = _locate_content_root(temp_dir)
        print(f"Using content from: {os.path.relpath(content_root, temp_dir)}")

        # Find OPF file
        opf_path = _find_opf_file(content_root)
        if not opf_path:
            raise ValueError("Could not find OPF file in MOBI extraction")

        # Extract metadata
        metadata = _extract_mobi_metadata(opf_path)

        # Prepare output directories
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Extract images
        print("Extracting images...")
        image_map = _extract_mobi_images(content_root, output_dir)

        # Extract spine order
        spine_items = _extract_spine_order(opf_path, content_root)

        # Parse TOC
        print("Parsing Table of Contents...")
        ncx_path = _find_ncx_file(content_root)
        if ncx_path:
            toc_structure = _parse_ncx_toc(ncx_path, content_root)
        else:
            print("Warning: No NCX file found, building fallback TOC from spine...")
            toc_structure = _build_fallback_mobi_toc(spine_items)

        # Process chapters
        print("Processing chapters...")
        spine_chapters = []

        for i, (href, item_id) in enumerate(spine_items):
            # Construct full path to content file
            content_file = os.path.join(content_root, href)

            if not os.path.exists(content_file):
                print(f"Warning: Content file not found: {href}")
                continue

            # Read HTML content
            try:
                with open(content_file, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_content = f.read()
            except Exception as e:
                print(f"Warning: Could not read {href}: {e}")
                continue

            soup = BeautifulSoup(raw_content, 'html.parser')

            # Fix image paths
            for img in soup.find_all('img'):
                src = img.get('src', '')
                if not src:
                    continue

                # Decode URL
                src_decoded = unquote(src)
                filename = os.path.basename(src_decoded)

                # Try to find in map
                if src_decoded in image_map:
                    img['src'] = image_map[src_decoded]
                elif filename in image_map:
                    img['src'] = image_map[filename]
                else:
                    # Try relative path from current file
                    rel_path = os.path.normpath(os.path.join(os.path.dirname(href), src_decoded))
                    rel_path = rel_path.replace('\\', '/')
                    if rel_path in image_map:
                        img['src'] = image_map[rel_path]

            # Clean HTML
            soup = clean_html_content(soup)

            # Extract body content
            body = soup.find('body')
            if body:
                final_html = "".join([str(x) for x in body.contents])
            else:
                final_html = str(soup)

            # Create ChapterContent
            chapter = ChapterContent(
                id=item_id,
                href=href,  # This links TOC to content
                title=f"Section {i+1}",
                content=final_html,
                text=extract_plain_text(soup),
                order=i
            )
            spine_chapters.append(chapter)

        # Assemble final book
        final_book = Book(
            metadata=metadata,
            spine=spine_chapters,
            toc=toc_structure,
            images=image_map,
            source_file=os.path.basename(mobi_path),
            processed_at=datetime.now().isoformat()
        )

        return final_book

    finally:
        # Clean up temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                print(f"Warning: Could not clean up temp directory: {e}")


# --- CLI ---

if __name__ == "__main__":

    import sys
    if len(sys.argv) < 2:
        print("Usage: python reader3.py <file.epub|file.mobi>")
        sys.exit(1)

    input_file = sys.argv[1]
    assert os.path.exists(input_file), "File not found."

    # Detect format by extension
    _, ext = os.path.splitext(input_file.lower())
    out_dir = os.path.splitext(input_file)[0] + "_data"

    # Route to appropriate processor
    if ext == '.epub':
        book_obj = process_epub(input_file, out_dir)
    elif ext == '.mobi':
        book_obj = process_mobi(input_file, out_dir)
    else:
        print(f"Unsupported format: {ext}")
        print("Supported formats: .epub, .mobi")
        sys.exit(1)

    save_to_pickle(book_obj, out_dir)
    print("\n--- Summary ---")
    print(f"Title: {book_obj.metadata.title}")
    print(f"Authors: {', '.join(book_obj.metadata.authors)}")
    print(f"Physical Files (Spine): {len(book_obj.spine)}")
    print(f"TOC Root Items: {len(book_obj.toc)}")
    print(f"Images extracted: {len(book_obj.images)}")
