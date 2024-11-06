import os
import subprocess

# Paths to the scripts
data_script = 'scripts/data_arxiv.py'
preprocess_script = 'scripts/preprocess_data.py'
train_script = 'scripts/train_bert.py'
evaluate_script = 'scripts/evaluate_model.py'


def run_script(script_path):
    """Helper function to run a Python script."""
    print(f"\nRunning {script_path}...")
    result = subprocess.run([
        'python', script_path], capture_output=True, text=True)
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

    # Step 3: Train the BERT model
    if os.path.exists(train_script):
        run_script(train_script)
    else:
        print(f"{train_script} not found. Please ensure the file exists.")

    # Step 4: Evaluate the trained model
    if os.path.exists(evaluate_script):
        run_script(evaluate_script)
    else:
        print(f"{evaluate_script} not found. Please ensure the file exists.")


if __name__ == '__main__':
    main()
