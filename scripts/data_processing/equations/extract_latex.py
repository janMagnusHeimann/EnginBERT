import os
import tarfile


def extract_latex_files(tarball_path, extract_dir="data/latex_extracted"):
    """
    Extracts LaTeX files from a tarball.
    
    Args:
        tarball_path (str): Path to the tarball file.
        extract_dir (str): Directory to extract the files.
    
    Returns:
        str: Path to the directory containing the extracted files.
    """
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tarball_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
    print(f"Extracted LaTeX files to {extract_dir}")
    return extract_dir

