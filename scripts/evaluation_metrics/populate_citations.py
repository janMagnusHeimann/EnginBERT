# populate_citations.py
import pandas as pd

# Load your processed dataset
df = pd.read_csv('data/cleaned_processed_papers.csv')

# Citation data for each document (replace this with actual citation data)
# Here, each document title is mapped to a list of titles it references
citations_data = {
    "Title of Document 1": [
        "Title of Cited Paper A", "Title of Cited Paper B"],
    "Title of Document 2": [
        "Title of Cited Paper C"],
    # Add more entries for each document title in df
}

# Populate 'citation_references' column based on the title
df["citation_references"] = df[
    "title"].apply(lambda x: citations_data.get(x, []))

print(df[["title", "citation_references"]].head())


# Save the updated dataset
df.to_csv('data/cleaned_processed_papers_with_citations.csv', index=False)
print("Citation references populated and " +
      "saved to 'data/cleaned_processed_papers_with_citations.csv'")
