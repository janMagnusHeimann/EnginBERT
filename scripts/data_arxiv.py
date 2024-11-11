import urllib.request
import xml.etree.ElementTree as ET
import pandas as pd
import os
import re
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError

# Directory to save data
os.makedirs("data", exist_ok=True)

# Define category-specific queries and labels
category_queries = {
    "Aerospace Engineering": (
        "cat:cs.AI OR cat:cs.RO", 1),
    "Mechanical Engineering": (
        "cat:cond-mat.mtrl-sci OR cat:cond-mat.stat-mech", 2),
    "Materials Science": (
        "cat:cond-mat.mtrl-sci OR cat:physics.chem-ph", 3)
}


def query_arxiv(search_query, start_index=0, max_results=10):
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


def parse_arxiv_data(data, label):
    papers = []
    namespace = {'atom': 'http://www.w3.org/2005/Atom'}

    root = ET.fromstring(data)
    for entry in root.findall('atom:entry', namespace):
        title = entry.find('atom:title', namespace).text
        abstract = entry.find('atom:summary', namespace).text
        pdf_url = None
        for link in entry.findall('atom:link', namespace):
            if link.attrib.get('title') == 'pdf':
                pdf_url = link.attrib['href']
                break

        papers.append({'title': title,
                       'abstract': abstract,
                       'pdf_url': pdf_url,
                       'labels': label})
    print(f"Parsed {len(papers)} papers for label {label}.")
    return papers


def extract_references_section(text):
    """Extract the References section from text by looking for the keyword 'references'."""
    start_idx = text.lower().find("references")
    if start_idx != -1:
        return text[start_idx:]
    return text  # Fallback to entire text if "References" not found


def extract_citation_titles(text):
    """Extracts probable titles from citations in the References section."""
    # Enhanced pattern to better match author-year-title structure
    citation_pattern = re.compile(
        r"([A-Z][a-zA-Z]+(?:, [A-Z]\.)?(?: and [A-Z][a-zA-Z]+)?(?:, [A-Z]\.)?)\s*\((\d{4})\)\.?\s*(.+?)\.\s*(?:[A-Z][a-z]+|Journal|Proceedings|In|Vol|pp|\n|$)",
        re.MULTILINE | re.DOTALL
    )

    citations = []
    for match in citation_pattern.finditer(text):
        author = match.group(1).strip()
        year = match.group(2).strip()
        title = match.group(3).strip()
        if len(title.split()) > 3:  # Basic filter for plausible titles
            citations.append(f"{author} ({year}). {title}")
 
    return citations


def download_and_extract_text_and_titles(pdf_url):
    file_name = 'temp_paper.pdf'
    try:
        urllib.request.urlretrieve(pdf_url, file_name)
        text = extract_text(file_name)
        os.remove(file_name)

        references_text = extract_references_section(text)
        citation_titles = extract_citation_titles(references_text)

        return text.strip(), citation_titles
    except (Exception, PDFSyntaxError) as e:
        print(f"Failed to download or extract text from {pdf_url}: {e}")
        return "", []


def collect_and_save_data():
    all_papers = []
    for category, (search_query, label) in category_queries.items():
        print(f"Collecting papers for category: {category}")
        data = query_arxiv(search_query, max_results=10)
        if data:
            papers = parse_arxiv_data(data, label)
            for paper in papers:
                full_text, citation_titles = download_and_extract_text_and_titles(paper['pdf_url'])
                paper['full_text'] = full_text
                paper['citation_references'] = citation_titles

            all_papers.extend(papers)

    # Save to CSV
    df = pd.DataFrame(all_papers)
    df.to_csv('data/processed_papers_with_citations.csv', index=False)
    print("Data collected and" +
          "saved to 'data/processed_papers_with_citations.csv'")

    # Verification step:
    # Print the first 10 entries with extracted citation titles
    print("\nFirst 10 entries with extracted citation titles:\n")
    for i in range(min(10, len(df))):
        title = df.loc[i, 'title']
        citations = df.loc[i, 'citation_references']
        print(f"Document {i+1} Title: {title}")
        print(f"Extracted Citation Titles: {citations}\n")


if __name__ == '__main__':
    collect_and_save_data()
