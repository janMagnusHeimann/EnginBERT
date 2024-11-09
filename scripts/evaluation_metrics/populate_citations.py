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
print("Citation references populated and saved to 'data/cleaned_processed_papers_with_citations.csv'")

# import pandas as pd

# # Load your processed dataset that contains document titles, abstracts, etc.
# # Ensure this file includes a 'title' column for document titles
# df = pd.read_csv("data/cleaned_processed_papers.csv")

# # Dictionary mapping each document title to a list of titles it cites.
# # Replace this with actual citation data for each document in your dataset.
# # The keys are document titles, and the values are lists of cited document titles.
# citations_data = {
#     "A Human - machine interface for teleoperation ...": [
#         "Artificial Intelligence and Systems Theory: Ap...",
#         "Topological Navigation of Simulated Robots usi...",
#     ],
#     "Safe cooperative robot dynamics on graphs": [
#         "A Human - machine interface for teleoperation ...",
#         "Robust Global Localization Using Clustered Par...",
#     ],
#     # Add more mappings here as needed to populate actual citations for all documents
# }

# # Populate the 'citation_references' column based on the title
# # This will create a list of citations for each document using citations_data
# df["citation_references"] = df["title"].apply(lambda x: citations_data.get(x, []))

# # Save the updated DataFrame with citation references to a new CSV
# # This file is used for citation retrieval evaluation, where 'citation_references'
# # provides the ground truth for each document's cited works
# df.to_csv("data/cleaned_processed_papers_with_citations.csv", index=False)
# print("Citation references populated and saved to 'data/cleaned_processed_papers_with_citations.csv'")
