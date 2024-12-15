import json
import os


def save_extracted_data(extracted_data, output_file="data/extracted_equations.json"):
    """
    Saves extracted equations and context to a JSON file.
    
    Args:
        extracted_data (list): List of dictionaries with "equation" and "context".
        output_file (str): Path to the output JSON file.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, ensure_ascii=False, indent=4)
    print(f"Saved extracted data to {output_file}")
