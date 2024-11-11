import os
import subprocess

# Paths to the scripts
data_script = "scripts/data_arxiv.py"
preprocess_script = "scripts/preprocess_data.py"
mlm_training_script = "scripts/train/mlm_training.py"
sequence_classification_script = "scripts/train/" \
                                 "train_bert_sequence_classification.py"
embedding_extraction_script = "scripts/embedding_extraction.py"
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

    # Step 3: Fine-tune BERT with Masked Language Modeling (MLM)
    if os.path.exists(mlm_training_script):
        run_script(mlm_training_script)
    else:
        print(f"{mlm_training_script} not found. " +
              "Please ensure the file exists.")

    # Step 4: Train Sequence Classification Model using Fine-Tuned Embeddings
    if os.path.exists(sequence_classification_script):
        run_script(sequence_classification_script)
    else:
        print(f"{sequence_classification_script} not found." +
              " Please ensure the file exists.")

    # Step 6: Extract embeddings from the fine-tuned model
    if os.path.exists(embedding_extraction_script):
        run_script(embedding_extraction_script)
    else:
        print(f"{embedding_extraction_script} not found. " +
              "Please ensure the file exists.")

    # # Step 7: Evaluate category clustering
    # if os.path.exists(clustering_script):
    #     run_script(clustering_script)
    # else:
    #     print(f"{clustering_script} not found. " +
    #            "Please ensure the file exists.")

    # # Step 8: Evaluate citation retrieval
    # if os.path.exists(citation_script):
    #     run_script(citation_script)
    # else:
    #     print(f"{citation_script} not found. Please ensure the file exists.")

    # # Step 9: Evaluate information retrieval
    # if os.path.exists(ir_script):
    #     run_script(ir_script)
    # else:
    #     print(f"{ir_script} not found. Please ensure the file exists.")


if __name__ == "__main__":
    main()
