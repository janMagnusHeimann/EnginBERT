import urllib.request
import xml.etree.ElementTree as ET
import pandas as pd
import os
from pdfminer.high_level import extract_text

# Directory to save data
os.makedirs("data", exist_ok=True)

# Function to query arXiv API with a broader search (across all categories)
def query_arxiv(start_index=0, max_results=10):
    base_url = 'http://export.arxiv.org/api/query'
    query_params = {
        'search_query': 'all',  # Broad search across all categories
        'start': start_index,
        'max_results': max_results
    }
    query_string = urllib.parse.urlencode(query_params)
    url = f'{base_url}?{query_string}'

    with urllib.request.urlopen(url) as response:
        data = response.read()
    return data

# Function to parse XML and extract paper information
def parse_arxiv_data(data):
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
        label = 0 if 'physics' in title.lower() else 1  # Simple labeling: 0 for Physics, 1 for Engineering
        papers.append({'title': title, 'abstract': abstract, 'pdf_url': pdf_url, 'labels': label})

    print(f"Parsed {len(papers)} papers.")
    return papers

# Function to download PDF and extract text
def download_and_extract_text(pdf_url):
    file_name = 'temp_paper.pdf'
    try:
        urllib.request.urlretrieve(pdf_url, file_name)
        text = extract_text(file_name)
        os.remove(file_name)  # Clean up downloaded PDF
        return text.strip()
    except Exception as e:
        print(f"Failed to download or extract text from {pdf_url}: {e}")
        return ""

# Collect and save papers to CSV with 'full_text' column
def collect_and_save_data(start_index=0, max_results=10):
    data = query_arxiv(start_index, max_results)
    papers = parse_arxiv_data(data)

    if papers:
        # Convert to DataFrame and add full_text
        df = pd.DataFrame(papers)
        df['full_text'] = df['pdf_url'].apply(download_and_extract_text)
        df.to_csv('data/processed_papers.csv', index=False)
        print("Data collected and saved to 'data/processed_papers.csv'")
    else:
        print("No data collected.")

# Main function to execute data collection
if __name__ == '__main__':
    collect_and_save_data(start_index=0, max_results=3)  # Adjust max_results as needed

