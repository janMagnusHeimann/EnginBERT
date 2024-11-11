import urllib.request
import xml.etree.ElementTree as ET
import pandas as pd
import re
import os
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError

# Directories to save data and PDFs
os.makedirs("data", exist_ok=True)
os.makedirs("data/pdfs", exist_ok=True)

# Define search query parameters
search_query = "cat:cs.AI"  # Example category for AI-related papers
max_results = 1  # Adjust to retrieve more papers if needed

def query_arxiv(search_query, start_index=0, max_results=1):
    """Query the arXiv API and retrieve metadata for a given search query."""
    base_url = 'http://export.arxiv.org/api/query'
    query_params = {
        'search_query': search_query,
        'start': start_index,
        'max_results': max_results
    }
    query_string = urllib.parse.urlencode(query_params)
    url = f'{base_url}?{query_string}'

    try:
        with urllib.request.urlopen(url) as response:
            data = response.read()
        return data
    except Exception as e:
        print(f"Failed to query arXiv: {e}")
        return None

def parse_arxiv_data(data):
    """Parse XML data from arXiv API and retrieve title, abstract, and PDF URL."""
    papers = []
    namespace = {'atom': 'http://www.w3.org/2005/Atom'}
    root = ET.fromstring(data)
    
    # Parse the first entry
    entry = root.find('atom:entry', namespace)
    title = entry.find('atom:title', namespace).text
    abstract = entry.find('atom:summary', namespace).text
    pdf_url = None
    for link in entry.findall('atom:link', namespace):
        if link.attrib.get('title') == 'pdf':
            pdf_url = link.attrib['href']
            break
    
    # Return metadata
    papers.append({'title': title, 'abstract': abstract, 'pdf_url': pdf_url})
    print(f"Retrieved paper: {title}")
    return papers[0]

def download_and_extract_text(pdf_url, save_filename):
    """Download a PDF from a URL and extract its full text."""
    try:
        urllib.request.urlretrieve(pdf_url, save_filename)
        print(f"PDF saved as '{save_filename}'")
        
        # Extract text using pdfminer
        text = extract_text(save_filename)
        return clean_text(text)
    except (Exception, PDFSyntaxError) as e:
        print(f"Failed to download or extract text from {pdf_url}: {e}")
        return ""

def clean_text(text):
    """Clean extracted text to remove unwanted artifacts and normalize content."""
    # Remove encoding artifacts and extra whitespace
    text = text.replace("(cid:12)", "-")
    text = text.replace("{", "").replace("}", "")
    text = text.replace("(cid:14)", "-").replace("(cid:11)", "-")  # Add more as needed
    
    # Remove line breaks within paragraphs, keeping true paragraph breaks
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)  # Removes line breaks within paragraphs
    text = re.sub(r'\n{2,}', '\n\n', text)  # Ensures paragraph separation
    return text.strip()

def extract_references_section(text):
    """Extract the References section from text by looking for the keyword 'references'."""
    start_idx = text.lower().find("references")
    if start_idx != -1:
        return text[start_idx:]
    return text  # Fallback to entire text if "References" not found

def extract_citation_titles(text):
    """Extract citations from the References section, capturing author, year, and title."""
    citations = []
    # Enhanced pattern to capture multi-line citations
    pattern = r"((?:[A-Z][a-z]+(?:, [A-Z]\.)?(?: and [A-Z][a-z]+)?(?:, [A-Z]\.)?)|(?:[A-Z][a-z]+ et al\.)),?\s*\((\d{4})\)\.?\s*(.+?)\.\s*(?:In|Proceedings|Journal|Tech|Vol|pp|\n|$)"
    
    for match in re.finditer(pattern, text, re.MULTILINE | re.DOTALL):
        author = match.group(1).strip()
        year = match.group(2).strip()
        title = match.group(3).strip()
        citation = f"{author} ({year}). {title}"
        # Filter to avoid incomplete titles
        if len(title.split()) > 3:
            citations.append(citation)
    return citations

def contains_valid_citation_title(citations):
    """Check if there is at least one valid citation title in the extracted citations."""
    return any(len(citation.split()) > 3 for citation in citations)

def process_single_paper():
    """Process a single arXiv paper: download, extract text, identify references."""
    data = query_arxiv(search_query, max_results=max_results)
    if data is None:
        print("No data retrieved from arXiv.")
        return
    
    paper = parse_arxiv_data(data)
    sanitized_title = re.sub(r'[\\/*?:"<>|]', '', paper['title'])  # Sanitize filename
    pdf_filename = f"data/pdfs/{sanitized_title}.pdf"

    # Download PDF and extract full text
    full_text = download_and_extract_text(paper['pdf_url'], pdf_filename)
    if not full_text:
        print("No full text extracted.")
        return

    # Extract references section and citation titles
    references_text = extract_references_section(full_text)
    citation_titles = extract_citation_titles(references_text)
    
    # Prepare final data for saving
    paper['full_text'] = full_text
    paper['citation_references'] = citation_titles

    # Save to CSV
    df = pd.DataFrame([paper])
    df.to_csv('data/single_processed_paper_with_citations.csv', index=False)
    print("Single data entry saved to 'data/single_processed_paper_with_citations.csv'")

    # Output for verification
    print("\nExtracted Citation References:")
    for citation in citation_titles:
        print(f"- {citation}")
    has_valid_citation_titles = contains_valid_citation_title(citation_titles)
    print(f"\nContains Valid Citation Titles: {'Yes' if has_valid_citation_titles else 'No'}")


if __name__ == '__main__':
    process_single_paper()
