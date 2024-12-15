import os
from scripts.data_processing.equations.download_latex import download_latex_tarball
from scripts.data_processing.equations.extract_latex import extract_latex_files
from scripts.data_processing.equations.extract_equations import extract_equations_and_context
from scripts.data_processing.equations.save_extracted_data import save_extracted_data
from scripts.data_processing.equations.category_crawler import query_arxiv_by_category


# eqation extraction, doesn't work flawlessly, doesn't always find end of equation.
# One might have to switch to API to get the data.


def main(arxiv_ids=None, categories=None, context_window=300, output_file="data/extracted_equations.json", max_results_per_category=10):
    """
    Main pipeline for downloading, extracting, and saving equations with context.
    
    Args:
        arxiv_ids (list): List of arXiv IDs to process.
        categories (list): List of arXiv categories to query. If provided, 
                           the script will fetch papers from these categories 
                           and extract their IDs.
        context_window (int): Number of characters for equation context.
        output_file (str): Path to save the final JSON file.
        max_results_per_category (int): Number of papers to fetch per category.
    """
    if arxiv_ids is None:
        arxiv_ids = []

    # If categories are provided, query each category and gather arXiv IDs
    if categories:
        for category in categories:
            papers = query_arxiv_by_category(category, max_results=max_results_per_category)
            # Extract IDs from the returned papers
            category_ids = []
            for paper in papers:
                # The arXiv ID in the feed is typically in the entry.id, which looks like:
                # "http://arxiv.org/abs/2204.04521v1"
                # We need to extract just the arXiv ID part.
                arxiv_id = paper["id"].split('/')[-1]
                # Some IDs have a version number (e.g., 2204.04521v1). 
                # You might want to strip off the version part:
                arxiv_id = arxiv_id.split('v')[0]
                category_ids.append(arxiv_id)
            
            arxiv_ids.extend(category_ids)

    # Remove duplicates if any
    arxiv_ids = list(set(arxiv_ids))

    all_extracted_data = []
    for arxiv_id in arxiv_ids:
        # Step 1: Download LaTeX tarball
        tarball_path = download_latex_tarball(arxiv_id)
        if not tarball_path:
            continue

        # Step 2: Extract LaTeX files
        extract_dir = extract_latex_files(tarball_path)

        # Step 3: Extract equations and context
        for root, _, files in os.walk(extract_dir):
            for file in files:
                if file.endswith(".tex"):
                    tex_path = os.path.join(root, file)
                    extracted_data = extract_equations_and_context(tex_path, context_window)
                    all_extracted_data.extend(extracted_data)
    
    # Step 4: Save extracted data
    save_extracted_data(all_extracted_data, output_file)


if __name__ == "__main__":
    # Example usage:
    # Option 1: Provide explicit arXiv IDs
    arxiv_ids = ["2204.04521"]
    # Option 2: Provide categories to fetch IDs dynamically
    # For example, "cs.LG" for Machine Learning (in Computer Science)
    categories = ["cs.LG"]

    # If you want to use categories, you can leave arxiv_ids empty or provide both
    #arxiv_ids = []

    main(
        arxiv_ids=arxiv_ids,
        categories=categories,
        context_window=300, 
        output_file="data/extracted_equations.json",
        max_results_per_category=5
    )
