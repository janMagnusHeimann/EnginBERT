# populate_citations.py

import pandas as pd
import re
from fuzzywuzzy import process

# Load the dataset
df = pd.read_csv('data/processed_papers.csv')

# Clean the 'title' column by removing newlines, extra spaces, and standardizing to lowercase
def clean_title(title):
    return re.sub(r'\s+', ' ', title.replace('\n', ' ')).strip().lower()

df["title"] = df["title"].apply(clean_title)

# Example citation data for demonstration
citations_data = {
    "a human - machine interface for teleoperation of arm manipulators in a complex environment": [
        "Title of Cited Paper A", "Title of Cited Paper B"],
    "safe cooperative robot dynamics on graphs": [
        "Title of Cited Paper C"],
    # Add more entries for each document title in df
}

# Clean and standardize titles in citations_data
citations_data = {clean_title(k): [clean_title(citation) for citation in v] for k, v in citations_data.items()}


# Function to use fuzzy matching if exact title match is not found
def find_citations(title, citations_dict, threshold=80):
    # Check for an exact match first
    if title in citations_dict:
        return citations_dict[title]
 
    # Fuzzy match if no exact match is found
    match, score = process.extractOne(title, citations_dict.keys())
    if score >= threshold:
        return citations_dict[match]
    return []


# Populate 'citation_references' column based on the title
df["citation_references"] = df["title"].apply(lambda x: find_citations(x, citations_data))

# Debugging step: Print unmatched titles
unmatched_titles = df[df["citation_references"].apply(len) == 0]["title"]
print("Unmatched titles:", unmatched_titles.tolist())

# Preview the result
print(df[["title", "citation_references"]].head())

# Save the updated dataset
df.to_csv('data/processed_papers_with_citations.csv', index=False)
print("Citation references populated and saved to 'data/processed_papers_with_citations.csv'")
