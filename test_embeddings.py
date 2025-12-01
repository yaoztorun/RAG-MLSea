from SPARQLWrapper import SPARQLWrapper, JSON
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import numpy as np
import json
import os

# ---------------------------
# 1) SETUP SPARQL ENDPOINT
# ---------------------------

endpoint_url = "http://localhost:7200/repositories/thesis"

query = """
PREFIX mlso: <http://w3id.org/mlso/>
PREFIX dcterms: <http://purl.org/dc/terms/>

SELECT ?paper ?title
WHERE {
  ?paper a mlso:ScientificWork ;
         dcterms:title ?title .
}
LIMIT 20
"""

print("Querying GraphDB...")

sparql = SPARQLWrapper(endpoint_url)
sparql.setReturnFormat(JSON)
sparql.setQuery(query)

result = sparql.query().convert()

bindings = result["results"]["bindings"]

titles = [b["title"]["value"] for b in bindings]
paper_iris = [b["paper"]["value"] for b in bindings]

print(f"Retrieved {len(titles)} paper titles")
print("Example titles:")
for t in titles[:5]:
    print(" -", t)

# ---------------------------
# 2) EMBEDDINGS WITH SBERT
# ---------------------------

print("\nLoading SBERT model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding titles...")
embeddings = model.encode(titles, show_progress_bar=True)

print("\nShape of embeddings:", embeddings.shape)
print("Example embedding for first title:\n", embeddings[0])

print("\nTitle 0:", titles[0])
for i in range(1, 5):
    sim = cos_sim(embeddings[0], embeddings[i])[0][0].item()
    print(f"Similarity with title {i}: {sim:.3f} -> {titles[i]}")

# ---------------------------
# 3) SAVE EMBEDDINGS + METADATA
# ---------------------------

output_dir = "stored_embeddings"
os.makedirs(output_dir, exist_ok=True)

# Save embeddings as .npy (fast, compact)
embeddings_path = os.path.join(output_dir, "paper_title_embeddings.npy")
np.save(embeddings_path, embeddings)
print(f"\nSaved embeddings to: {embeddings_path}")

# Save metadata as JSON
metadata = []
for iri, title in zip(paper_iris, titles):
    metadata.append({
        "paper_iri": iri,
        "title": title
    })

metadata_path = os.path.join(output_dir, "paper_title_metadata.json")
with open(metadata_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print(f"Saved metadata to: {metadata_path}")

# Quick check
loaded_embeddings = np.load("stored_embeddings/paper_title_embeddings.npy")
print("Loaded embeddings shape:", loaded_embeddings.shape)

with open("stored_embeddings/paper_title_metadata.json", "r", encoding="utf-8") as f:
    meta = json.load(f)
print("Loaded metadata entries:", len(meta))
print("Example metadata:", meta[0])


# ---------------------------
# 4) SEMANTIC SEARCH FUNCTION
# ---------------------------

def semantic_search(query: str, top_k: int = 5):
    """
    Given a natural language query string, compute its embedding,
    compare it to all paper title embeddings, and return the top_k
    most similar papers.
    """
    if not query.strip():
        print("Empty query, please type something.")
        return

    # 1) Embed the query using the same model
    query_emb = model.encode([query])  # shape: (1, 384)

    # 2) Compute cosine similarity between query and all paper embeddings
    # cos_sim returns a 1 x N tensor; we take [0] to get the vector
    scores = cos_sim(query_emb, embeddings)[0].cpu().numpy()

    # 3) Get indices of top_k highest scores
    top_k = min(top_k, len(titles))
    top_indices = np.argsort(-scores)[:top_k]

    print(f"\nTop {top_k} results for query: {query!r}")
    for rank, idx in enumerate(top_indices, start=1):
        title = titles[idx]
        iri = paper_iris[idx]
        score = scores[idx]
        print(f"{rank}. score={score:.3f}")
        print(f"   title: {title}")
        print(f"   IRI:   {iri}")


# Optional: simple interactive loop
if __name__ == "__main__":
    while True:
        user_query = input("\nEnter a search query (or 'exit' to quit): ")
        if user_query.lower() in ("exit", "quit"):
            break
        semantic_search(user_query, top_k=5)
