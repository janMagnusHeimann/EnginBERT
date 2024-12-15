import urllib.request
import xml.etree.ElementTree as ET

def query_arxiv_by_category(category, max_results=10, start=0):
    """
    Query the arXiv API for a given category using the same logic as data_arxiv.py.
    
    :param category: str, an arXiv category, e.g. 'cs.LG' or 'physics:hep-th'.
    :param max_results: int, number of results to return.
    :param start: int, offset for pagination.
    :return: list of dicts, each containing metadata for a retrieved paper.
    """
    base_url = 'http://export.arxiv.org/api/query'
    query_params = {
        'search_query': f'cat:{category}',
        'start': start,
        'max_results': max_results
    }
    query_string = urllib.parse.urlencode(query_params)
    url = f'{base_url}?{query_string}'

    try:
        with urllib.request.urlopen(url) as response:
            data = response.read()
    except Exception as e:
        print(f"Failed to query arXiv: {e}")
        return []

    return parse_arxiv_feed(data)

def parse_arxiv_feed(data):
    """
    Parse the ATOM feed returned by the arXiv API.
    
    :param data: XML data from arXiv.
    :return: list of paper metadata dicts.
    """
    namespace = {'atom': 'http://www.w3.org/2005/Atom'}
    root = ET.fromstring(data)

    papers = []
    for entry in root.findall('atom:entry', namespace):
        title = entry.find('atom:title', namespace)
        summary = entry.find('atom:summary', namespace)
        published = entry.find('atom:published', namespace)
        authors = entry.findall('atom:author', namespace)
        tags = entry.findall('atom:category', namespace)
        paper_id = entry.find('atom:id', namespace)

        paper_info = {
            "title": title.text.strip() if title is not None else None,
            "summary": summary.text.strip() if summary is not None else None,
            "id": paper_id.text.strip() if paper_id is not None else None,
            "published": published.text.strip() if published is not None else None,
            "authors": [a.find('atom:name', namespace).text.strip() 
                        for a in authors if a.find('atom:name', namespace) is not None],
            "categories": [t.attrib['term'] for t in tags if 'term' in t.attrib]
        }
        papers.append(paper_info)
    return papers