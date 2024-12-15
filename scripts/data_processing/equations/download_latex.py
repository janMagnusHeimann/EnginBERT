import os
import requests


def download_latex_tarball(arxiv_id, save_dir="data/latex_files"):
    """
    Downloads LaTeX tarball for a given arXiv ID.
    
    Args:
        arxiv_id (str): The arXiv ID of the paper.
        save_dir (str): Directory to save the LaTeX tarball.
    
    Returns:
        str: Path to the downloaded tarball or None if download fails.
    """
    base_url = f"https://arxiv.org/e-print/{arxiv_id}"
    response = requests.get(base_url)
       
    if response.status_code == 200:
        os.makedirs(save_dir, exist_ok=True)
        tarball_path = os.path.join(save_dir, f"{arxiv_id}.tar.gz")
        with open(tarball_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded LaTeX tarball for {arxiv_id}")
        return tarball_path
    else:
        print(f"Failed to download LaTeX tarball for {arxiv_id}")
        return None
