import re
import os


def extract_equations_and_context(tex_path, context_window=300):
    """
    Extracts equations and their surrounding context from a LaTeX file.
    
    Args:
        tex_path (str): Path to the LaTeX file.
        context_window (int): Number of characters around the equation to include as context.
    
    Returns:
        list: A list of dictionaries with "equation" and "context".
    """
    with open(tex_path, "r", encoding="utf-8") as f:
        latex_content = f.read()

    # Regex to find equations
    equation_pattern = re.compile(
        r"(\\begin{equation}.*?\\end{equation}|\\\[.*?\\\]|\\begin{align}.*?\\end{align})",
        re.DOTALL
    )

    matches = equation_pattern.finditer(latex_content)

    extracted_data = []
    for match in matches:
        equation = match.group(1)

        # Extract surrounding context
        start = max(match.start() - context_window, 0)
        end = min(match.end() + context_window, len(latex_content))
        context = latex_content[start:match.start()] + latex_content[match.end():end]

        # Clean LaTeX artifacts from context
        context = re.sub(r"\\[a-zA-Z]+", "", context)  # Remove LaTeX commands
        context = re.sub(r"\s+", " ", context).strip()  # Normalize spaces

        extracted_data.append({"equation": equation.strip(), "context": context})
    
    print(f"Extracted {len(extracted_data)} equations from {tex_path}")
    return extracted_data
