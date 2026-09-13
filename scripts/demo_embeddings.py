"""Demonstrate that embeddings capture meaning rather than shared words."""

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

phrases = [
    "delete empty rows",
    "drop missing values",
    "plot a bar chart",
]

vectors = model.encode(phrases, normalize_embeddings=True)

print(f"each phrase became {vectors.shape[1]} numbers\n")
print("first 8 numbers of 'delete empty rows':")
print(vectors[0][:8], "\n")

print("closeness (1.0 = identical meaning, 0.0 = unrelated):")
for i in range(len(phrases)):
    for j in range(i + 1, len(phrases)):
        score = float(vectors[i] @ vectors[j])
        print(f"  {score:.2f}   '{phrases[i]}'  vs  '{phrases[j]}'")
