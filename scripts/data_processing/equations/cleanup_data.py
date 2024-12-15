import os
import shutil

# Define paths to clean
DATA_DIRECTORIES = [
    "data/latex_files",
    "data/latex_extracted",
    "data",
]

MODEL_DIRECTORIES = [
    "model/bert_classification_model",
    "model/technical_term_model",
    "model/equation_model",
]

DATA_FILES = [
    "data/processed_papers_with_citations.csv",
    "data/cleaned_processed_papers.csv",
    "data/evaluation_processed_papers.csv",
    "data/extracted_equations.json",
]

def delete_directory(path):
    """Delete a directory and all its contents."""
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Deleted directory: {path}")
    else:
        print(f"Directory does not exist: {path}")

def delete_file(path):
    """Delete a single file."""
    if os.path.exists(path):
        os.remove(path)
        print(f"Deleted file: {path}")
    else:
        print(f"File does not exist: {path}")

def cleanup_data():
    """Remove all data directories and files."""
    print("Cleaning up data directories and files...")
    for dir_path in DATA_DIRECTORIES:
        delete_directory(dir_path)

    for file_path in DATA_FILES:
        delete_file(file_path)

def cleanup_models():
    """Remove all model directories."""
    print("Cleaning up model directories...")
    for dir_path in MODEL_DIRECTORIES:
        delete_directory(dir_path)

if __name__ == "__main__":
    print("WARNING: This will delete all generated data and models.")
    confirm = input("Type 'yes' to proceed: ").strip().lower()
    if confirm == "yes":
        cleanup_data()
        cleanup_models()
        print("Cleanup completed successfully.")
    else:
        print("Cleanup aborted.")
