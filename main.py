import os
import subprocess

# Paths to the scripts
data_script = "scripts/data_arxiv.py"
preprocess_script = "scripts/preprocess_data.py"
populate_citations_script = "scripts/evaluation_metrics/populate_citations.py"
train_script = "scripts/train_bert.py"
mod_tok_script = "scripts/Model_and_tokenizer.py"
clustering_script = "scripts/evaluation_metrics/category_clustering.py"
citation_script = "scripts/evaluation_metrics/citation_evaluation.py"
ir_script = "scripts/evaluation_metrics/information_retrieval.py"


def run_script(script_path):
    """Helper function to run a Python script."""
    print(f"\nRunning {script_path}...")
    result = subprocess.run(
        ["python", script_path], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"Error running {script_path}:\n{result.stderr}")


def main():
    # Step 1: Collect data from arXiv
    if os.path.exists(data_script):
        run_script(data_script)
    else:
        print(f"{data_script} not found. Please ensure the file exists.")

    # Step 2: Preprocess the data
    if os.path.exists(preprocess_script):
        run_script(preprocess_script)
    else:
        print(f"{preprocess_script} not found. Please ensure the file exists.")

    # Step 3: Populate citation references
    # if os.path.exists(populate_citations_script):
    #     run_script(populate_citations_script)
    # else:
    #     print(f"{populate_citations_script} not found." +
    #           "Please ensure the file exists.")

    # Step 3: Train the BERT model
    if os.path.exists(train_script):
        run_script(train_script)
    else:
        print(f"{train_script} not found. Please ensure the file exists.")

    # Step 4: Load Model and Tokenizer
    if os.path.exists(mod_tok_script):
        run_script(mod_tok_script)
    else:
        print(f"{mod_tok_script} not found. Please ensure the file exists.")

    # Step 5: Evaluate category clustering
    if os.path.exists(clustering_script):
        run_script(clustering_script)
    else:
        print(f"{clustering_script} not found. Please ensure the file exists.")

    # Step 6: Evaluate citation retrieval
    if os.path.exists(citation_script):
        run_script(citation_script)
    else:
        print(f"{citation_script} not found. Please ensure the file exists.")

    # Step 7: Evaluate information retrieval
    if os.path.exists(ir_script):
        run_script(ir_script)
    else:
        print(f"{ir_script} not found. Please ensure the file exists.")


if __name__ == "__main__":
    main()
