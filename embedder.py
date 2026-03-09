import json
import os
import time
import chromadb

from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import tiktoken


INPUT_FILE      = "papers.json"
CHROMA_DB_DIR   = "./chroma_db"          # where the vector DB is saved
COLLECTION_NAME = "arxiv_papers"

CHUNK_SIZE      = 500                    # tokens per chunk
CHUNK_OVERLAP   = 50                     # token overlap between chunks
BATCH_SIZE      = 32                     # embed N chunks at a time 
EMBED_MODEL     = "all-MiniLM-L6-v2"    


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Split text into overlapping token-based chunks.
    Uses tiktoken to count tokens accurately.
    """
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text, allowed_special="all")    
    chunks    = []
    
    start = 0
    while start < len(tokens):
        end        = min(start + chunk_size, len(tokens))
        chunk_toks = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_toks)
        chunks.append(chunk_text)

        # Move forward by (chunk_size - overlap) so chunks share context
        start += chunk_size - overlap

        # Stop if we've covered everything
        if end == len(tokens):
            break

    return chunks


def build_chunks(papers: list[dict]) -> list[dict]:
    """
    Takes the list of papers and returns a flat list of chunks,
    each tagged with metadata from its source paper.
    """
    all_chunks = []

    for paper in papers:
        text = paper.get("text", "").strip()
        if not text:
            continue

        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                # The actual text content
                "text": chunk,

                # Metadata stored alongside the vector in ChromaDB
                "metadata": {
                    "paper_title": paper["title"],
                    "authors":     ", ".join(paper["authors"][:3]),  # first 3 authors
                    "year":        str(paper["year"]),
                    "url":         paper["url"],
                    "topic":       paper.get("topic", ""),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },

                # Unique ID for ChromaDB (paper url + chunk index)
                "id": f"{paper['url'].split('/')[-1]}_chunk_{i}"
            })

    return all_chunks


#  STEP 2 — Load embedding model

def load_embedding_model(model_name: str) -> SentenceTransformer:
    """
    Load the HuggingFace embedding model.
    First run downloads it, after that it's cached locally.
    """
    print(f" Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"    Model loaded!\n")
    return model

#  STEP 3 — Embed chunks in batches

def embed_chunks(chunks: list[dict], model: SentenceTransformer, batch_size: int) -> list[dict]:
    """
    Embed all chunk texts in batches to avoid RAM overload.
    Adds embedding key to each chunk dict.
    """
    print(f" Embedding {len(chunks)} chunks in batches of {batch_size}...\n")

    texts      = [c["text"] for c in chunks]
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch     = texts[i : i + batch_size]
        batch_emb = model.encode(batch, show_progress_bar=False)
        embeddings.extend(batch_emb.tolist())

        # Progress indicator
        done = min(i + batch_size, len(texts))
        pct  = (done / len(texts)) * 100
        print(f"   [{done:>5}/{len(texts)}] {pct:5.1f}% done")

    # Attach embeddings back to chunk dicts
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb

    print(f"\n   All chunks embedded!\n")
    return chunks

#  STEP 4 — Store in ChromaDB

def store_in_chromadb(chunks: list[dict], db_dir: str, collection_name: str):
    """
    Store all embedded chunks in a persistent local ChromaDB.
    Creates the DB folder if it doesn't exist.
    """
    print(f" Storing in ChromaDB at '{db_dir}'...\n")

    # Create persistent client where data saved to disk automatically
    client     = chromadb.PersistentClient(path=db_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}   
    )

    # Insert in batches 
    INSERT_BATCH = 500
    for i in range(0, len(chunks), INSERT_BATCH):
        batch = chunks[i : i + INSERT_BATCH]

        collection.add(
            ids        = [c["id"]        for c in batch],
            documents  = [c["text"]      for c in batch],
            embeddings = [c["embedding"] for c in batch],
            metadatas  = [c["metadata"]  for c in batch],
        )

        done = min(i + INSERT_BATCH, len(chunks))
        print(f"   Inserted {done}/{len(chunks)} chunks...")

    print(f"\n   ChromaDB ready with {collection.count()} total chunks!\n")
    return collection

#  STEP 5 — Smoke test the retrieval

def smoke_test(collection, model: SentenceTransformer):
    """
    Run 3 test queries to confirm everything works.
    """
    test_queries = [
        "What is self-attention and how does it work?",
        "How does LoRA reduce GPU memory during fine-tuning?",
        "What is retrieval augmented generation?",
    ]

    print("=" * 60)
    print("  SMOKE TEST — checking retrieval works")
    print("=" * 60)

    for query in test_queries:
        query_embedding = model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=2,
            include=["documents", "metadatas", "distances"]
        )

        print(f"\n Query: {query}")
        for j, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            score = 1 - dist   
            print(f"\n  Result {j+1} (similarity: {score:.3f})")
            print(f"  {meta['paper_title'][:60]} ({meta['year']})")
            print(f"  {doc[:150].strip()}...")

    print("\n" + "=" * 60)
    print("  Smoke test passed! Ready for Phase 3.")
    print("=" * 60 + "\n")


#  MAIN PIPELINE

def main():
    # Load papers
    print(f" Loading papers from '{INPUT_FILE}'...\n")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        papers = json.load(f)
    print(f"   Loaded {len(papers)} papers\n")

    # Chunk
    print("  Chunking papers...\n")
    chunks = build_chunks(papers)
    print(f"   Created {len(chunks):,} chunks from {len(papers)} papers")

    # Stats breakdown
    avg_chunks = len(chunks) / len(papers)
    print(f"   Average chunks per paper: {avg_chunks:.1f}\n")

    # Embed
    model  = load_embedding_model(EMBED_MODEL)
    chunks = embed_chunks(chunks, model, BATCH_SIZE)

    # Store
    collection = store_in_chromadb(chunks, CHROMA_DB_DIR, COLLECTION_NAME)

    # Smoke test
    smoke_test(collection, model)

    # Final summary
    print(f"   -> {len(chunks):,} chunks stored in '{CHROMA_DB_DIR}/'")
    print(f"   -> Collection name: '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()