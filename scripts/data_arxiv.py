import urllib.request
import xml.etree.ElementTree as ET
import pandas as pd
import os
from pdfminer.high_level import extract_text

# Directory to save data
os.makedirs("data", exist_ok=True)

# Define category-specific queries and labels
category_queries = {
    "Physics": ("cat:physics", 0),
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

    with urllib.request.urlopen(url) as response:
        data = response.read()
    return data


# Parse XML and extract paper info
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

        papers.append({
            'title': title, 'abstract': abstract,
            'pdf_url': pdf_url, 'labels': label})

    print(f"Parsed {len(papers)} papers for label {label}.")
    return papers


# Download and extract full text from PDFs
def download_and_extract_text(pdf_url):
    file_name = 'temp_paper.pdf'
    try:
        urllib.request.urlretrieve(pdf_url, file_name)
        text = extract_text(file_name)
        os.remove(file_name)
        return text.strip()
    except Exception as e:
        print(f"Failed to download or extract text from {pdf_url}: {e}")
        return ""


# Collect data across categories and save
def collect_and_save_data():
    all_papers = []

    for category, (search_query, label) in category_queries.items():
        print(f"Collecting papers for category: {category}")
        data = query_arxiv(search_query, max_results=25)
        # Adjust max_results for each category as needed
        papers = parse_arxiv_data(data, label)

        # Add full text to each paper
        for paper in papers:
            paper['full_text'] = download_and_extract_text(paper['pdf_url'])

        all_papers.extend(papers)

    # Save to CSV
    df = pd.DataFrame(all_papers)
    df.to_csv('data/processed_papers.csv', index=False)
    print("Data collected and saved to 'data/processed_papers.csv'")


# Main function to execute data collection
if __name__ == '__main__':
    collect_and_save_data()
