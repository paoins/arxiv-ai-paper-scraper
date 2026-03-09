import os
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer


CHROMA_DB_DIR   = "./chroma_db"
COLLECTION_NAME = "arxiv_papers"
EMBED_MODEL     = "all-MiniLM-L6-v2"
GROQ_MODEL      = "llama-3.3-70b-versatile"   

TOP_K           = 5     # number of chunks to retrieve per query
MAX_HISTORY     = 5     # how many past messages to keep in context


SYSTEM_PROMPT = """You are an expert AI research assistant with deep knowledge of machine learning papers.

Your job is to answer questions using ONLY the research paper excerpts provided in each message.

Rules:
- Always ground your answer in the provided context
- Always mention which paper(s) your answer comes from
- If the context doesn't contain enough info, say so honestly
- Be precise and technical: the user is an AI/ML student
- When explaining complex concepts, use clear analogies
- Keep answers focused and well-structured

Format your answer like this:
1. Direct answer to the question
2. Key details from the paper(s)
3. Source: [Paper Title] ([Year]) — [ArXiv URL]
"""

SIMPLE_PROMPT = """You are a friendly AI tutor explaining research papers to a curious student.

Your job is to answer questions using ONLY the research paper excerpts provided.

Rules:
- Explain everything simply , no jargon without explanation
- Use real-world analogies to make concepts click
- Always mention which paper the explanation comes from
- If you don't know from the context, say so

Format:
1. Simple explanation (use an analogy)
2. A bit more detail
3. Source: [Paper Title] ([Year])
"""

COMPARE_PROMPT = """You are an expert AI researcher comparing different papers and approaches.

Your job is to compare and contrast ideas using ONLY the research paper excerpts provided.

Rules:
- Clearly identify which idea/approach belongs to which paper
- Highlight key similarities AND differences
- Be specific — use numbers/metrics from papers when available
- Always cite sources

Format:
1. Overview of each approach
2. Key differences
3. Which is better for what use case
4. Sources: [Paper 1] vs [Paper 2]
"""

#  STEP 1 — Load models and DB


def load_resources():
    """Load the embedding model, ChromaDB, and Groq client."""

    print(" Loading resources...\n")

    # Embedding model (same one used in Phase 2)
    print("  Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL)
    print("     Embedding model ready")

    # ChromaDB
    print("   Connecting to ChromaDB...")
    client     = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"     ChromaDB ready — {collection.count():,} chunks loaded")

    # Groq client
    print("  Connecting to Groq...")
    api_key = os.environ.get("GROQ_API_KEY")
    groq_client = Groq(api_key=api_key)
    print("      Groq client ready\n")

    return embed_model, collection, groq_client

#  STEP 2 — Retrieve relevant chunks

def retrieve(query: str, collection, embed_model, top_k: int = TOP_K) -> list[dict]:
    """
    Embed the query and find the most similar chunks in ChromaDB.
    Returns a list of chunks with text + metadata.
    """
    query_embedding = embed_model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":        doc,
            "title":       meta["paper_title"],
            "authors":     meta["authors"],
            "year":        meta["year"],
            "url":         meta["url"],
            "similarity":  round(1 - dist, 3),   # cosine similarity score
        })

    return chunks

#  STEP 3 — Format context for the prompt

def format_context(chunks: list[dict]) -> str:
    """Turn retrieved chunks into a clean context block for the LLM."""
    context_parts = []

    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"--- Excerpt {i+1} ---\n"
            f"Paper: {chunk['title']} ({chunk['year']})\n"
            f"Authors: {chunk['authors']}\n"
            f"URL: {chunk['url']}\n"
            f"Relevance Score: {chunk['similarity']}\n\n"
            f"{chunk['text']}\n"
        )

    return "\n".join(context_parts)

#  STEP 4 — Generate answer with Groq


def generate(
    query:       str,
    chunks:      list[dict],
    groq_client: Groq,
    history:     list[dict],
    mode:        str = "chat",
) -> str:
    """
    Send retrieved context + query to Groq LLaMA 3.3 and return the answer.
    Includes conversation history for multi-turn memory.
    """
    # Pick system prompt based on mode
    system_prompts = {
        "chat":    SYSTEM_PROMPT,
        "simple":  SIMPLE_PROMPT,
        "compare": COMPARE_PROMPT,
    }
    system = system_prompts.get(mode, SYSTEM_PROMPT)

    # Build user message with context
    context      = format_context(chunks)
    user_message = f"Context from research papers:\n\n{context}\n\nQuestion: {query}"

    # Build messages array: system + history + current message
    messages = (
        [{"role": "system", "content": system}]
        + history[-MAX_HISTORY:]                    # last N turns for memory
        + [{"role": "user", "content": user_message}]
    )

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.3,     # low = more factual, less hallucination
        max_tokens=1024,
    )

    return response.choices[0].message.content

#  STEP 5 — Display sources nicely

def display_sources(chunks: list[dict]):
    """Print the source papers used to answer the query."""
    # Deduplicate by paper title
    seen   = set()
    unique = []
    for c in chunks:
        if c["title"] not in seen:
            seen.add(c["title"])
            unique.append(c)

    print("\n" + "─" * 50)
    print("Sources used:")
    for c in unique:
        print(f"  • {c['title'][:55]} ({c['year']})")
        print(f"    {c['url']}")
        print(f"    Relevance: {c['similarity']}")
    print("─" * 50)

#  STEP 6 — Terminal chat loop

def print_banner():
    print("\n" + "=" * 60)
    print("  ArXiv AI Paper Assistant")
    print("  Powered by Llama 3.3 70B (Groq) + ChromaDB")
    print("=" * 60)
    print("\nModes (type to switch):")
    print("  /chat     — standard Q&A mode (default)")
    print("  /simple   — explain like I'm a student mode")
    print("  /compare  — compare two papers/approaches")
    print("  /clear    — clear conversation history")
    print("  /quit     — exit\n")
    print("Example questions:")
    print("  • How does self-attention work?")
    print("  • What problem does LoRA solve?")
    print("  • How does RAG reduce hallucinations?")
    print("  • Compare attention with convolution\n")
    print("─" * 60 + "\n")


def chat_loop(embed_model, collection, groq_client):
    """Main interactive terminal chat loop."""
    history = []
    mode    = "chat"

    mode_labels = {
        "chat":    "Chat",
        "simple":  "Explain Simply",
        "compare": "Compare",
    }

    while True:
        try:
            user_input = input(f"[{mode_labels[mode]}] You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n Goodbye!\n")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() == "/quit":
            print("\n Goodbye!\n")
            break
        elif user_input.lower() == "/clear":
            history = []
            print(" Conversation history cleared.\n")
            continue
        elif user_input.lower() == "/chat":
            mode = "chat"
            print(f" Switched to {mode_labels[mode]} mode\n")
            continue
        elif user_input.lower() == "/simple":
            mode = "simple"
            print(f" Switched to {mode_labels[mode]} mode\n")
            continue
        elif user_input.lower() == "/compare":
            mode = "compare"
            print(f" Switched to {mode_labels[mode]} mode\n")
            continue

        # Retrieve relevant chunks
        print("\n Searching papers...")
        chunks = retrieve(user_input, collection, embed_model)

        # Generate answer
        print(" Generating answer...\n")
        answer = generate(user_input, chunks, groq_client, history, mode)

        # Print answer
        print(f" Assistant:\n{answer}")

        # Show sources
        display_sources(chunks)
        print()

        # Update history (store clean question + answer without context)
        history.append({"role": "user",      "content": user_input})
        history.append({"role": "assistant", "content": answer})


#  MAIN

def main():
    print_banner()
    embed_model, collection, groq_client = load_resources()
    chat_loop(embed_model, collection, groq_client)


if __name__ == "__main__":
    main()