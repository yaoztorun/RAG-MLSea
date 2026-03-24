from SPARQLWrapper import SPARQLWrapper, JSON
from sentence_transformers import SentenceTransformer, CrossEncoder
from sentence_transformers.util import cos_sim
import numpy as np
from collections import defaultdict

ENDPOINT = "http://localhost:7200/repositories/thesis"

# Used a broad query to get a good chunk of papers to demonstrate ranking
QUERY_ALL_PAPERS = """
PREFIX mlso: <http://w3id.org/mlso/>
PREFIX dcterms: <http://purl.org/dc/terms/>
PREFIX dcat: <http://www.w3.org/ns/dcat#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?paper ?title ?year ?authorName ?taskLabel ?kw
WHERE {
  ?paper a mlso:ScientificWork ;
         dcterms:title ?title .
  OPTIONAL { ?paper dcat:keyword ?kw . }
  OPTIONAL { ?paper dcterms:issued ?year . }
  OPTIONAL { 
    ?paper dcterms:creator ?author . 
    OPTIONAL { ?author rdfs:label ?label . }
    BIND(COALESCE(?label, STR(?author)) AS ?authorName)
  }
  OPTIONAL {
    ?paper mlso:hasTaskType ?task .
    ?task rdfs:label ?taskLabel .
  }
}
LIMIT 1000
"""

def run_sparql(endpoint_url: str, query: str):
    sparql = SPARQLWrapper(endpoint_url)
    sparql.setReturnFormat(JSON)
    sparql.setQuery(query)
    return sparql.query().convert()["results"]["bindings"]

def build_paper_chunks(rows):
    titles = {}
    years = {}
    kws = defaultdict(set)
    authors = defaultdict(set)
    tasks = defaultdict(set)
    
    for r in rows:
        paper = r["paper"]["value"]
        title = r["title"]["value"]
        titles[paper] = title
        
        if "year" in r and "value" in r["year"]:
            years[paper] = r["year"]["value"]
        if "kw" in r and "value" in r["kw"]:
            kws[paper].add(r["kw"]["value"])
        if "authorName" in r and "value" in r["authorName"]:
            authors[paper].add(r["authorName"]["value"])
        if "taskLabel" in r and "value" in r["taskLabel"]:
            tasks[paper].add(r["taskLabel"]["value"])

    chunks = []
    for paper, title in titles.items():
        year_text = years.get(paper, "Unknown")
        kw_text = "; ".join(sorted(kws[paper])) if kws[paper] else "None"
        author_text = "; ".join(sorted(authors[paper])) if authors[paper] else "Unknown"
        task_text = "; ".join(sorted(tasks[paper])) if tasks[paper] else "None"
        
        chunk_text = (
            f"Title: {title}\n"
            f"Year: {year_text}\n"
            f"Authors: {author_text}\n"
            f"Tasks: {task_text}\n"
            f"Keywords: {kw_text}"
        )
        chunks.append({
            "paper_iri": paper,
            "title": title,
            "text": chunk_text
        })
    return chunks


# ==========================================
# POST-RETRIEVAL PIPELINE START
# ==========================================

def post_retrieval_pipeline(question, chunks, chunk_embeddings, bi_encoder, cross_encoder):
    """
    Step 1: Fast SBERT Retrieval (fetch Top-20 candidates)
    Step 2: Score Hard Filter 
    Step 3: Cross-Encoder Re-Ranking
    Step 4: Top-K Context Formatting for LLM
    """
    print(f"\n--- Processing Question: '{question}' ---")
    
    # 1. BI-ENCODER RETRIEVAL (Get Top 20)
    q_emb = bi_encoder.encode([question])
    sbert_scores = cos_sim(q_emb, chunk_embeddings)[0].cpu().numpy()
    
    top_20_idx = np.argsort(-sbert_scores)[:20]
    retrieved_candidates = [(float(sbert_scores[i]), chunks[i]) for i in top_20_idx]

    # 2. HARD FILTERING (Drop anything below SBERT score of 0.20)
    filtered_candidates = [c for c in retrieved_candidates if c[0] > 0.20]
    
    if not filtered_candidates:
        return "The question is unanswerable from the available MLSea knowledge graph context."

    # 3. CROSS-ENCODER RE-RANKING
    # We pair the question with each retrieved text: [[Q, Text1], [Q, Text2], ...]
    cross_inputs = [[question, c[1]["text"]] for c in filtered_candidates]
    ce_scores = cross_encoder.predict(cross_inputs)
    
    # Attach the new Cross-Encoder scores to the candidates and re-sort
    for idx, cand in enumerate(filtered_candidates):
        cand[1]["cross_score"] = float(ce_scores[idx])
        
    # Sort strictly by the cross-encoder score
    reranked_candidates = sorted(filtered_candidates, key=lambda x: x[1]["cross_score"], reverse=True)

    # 4. TOP-K SELECTION & FORMATTING
    # We only pass the absolute best 3 results to the LLM to prevent hallucinations
    top_3_final = reranked_candidates[:3]
    
    context_string = "<CONTEXT>\n"
    for rank, cand in enumerate(top_3_final, 1):
        chunk = cand[1]
        context_string += f"--- Result {rank} ---\n"
        context_string += f"Entity IRI: {chunk['paper_iri']}\n"
        context_string += f"{chunk['text']}\n\n"
        
    context_string += "</CONTEXT>"
    
    return context_string

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    print("1. Fetching up to 100 papers from GraphDB...")
    try:
        rows = run_sparql(ENDPOINT, QUERY_ALL_PAPERS)
        chunks = build_paper_chunks(rows)
        print(f"   -> Built {len(chunks)} contextual chunks.")
    except Exception as e:
        print(f"Could not connect to GraphDB at {ENDPOINT}. Is it running?\nError: {e}")
        return

    print("2. Loading SBERT (Bi-Encoder) and passing chunks...")
    bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [c["text"] for c in chunks]
    chunk_embeddings = bi_encoder.encode(texts)

    print("3. Loading MS-MARCO Cross-Encoder (Re-Ranker)...")
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # Test our pipeline!
    test_question = "Which papers discuss the concept of contrastive learning?"
    
    # Run the Post-Retrieval Pipeline
    final_llm_context = post_retrieval_pipeline(test_question, chunks, chunk_embeddings, bi_encoder, cross_encoder)
    
    print("\n================ FINAL OUTPUT TO SEND TO LLM ================\n")
    print(final_llm_context)

if __name__ == "__main__":
    main()
