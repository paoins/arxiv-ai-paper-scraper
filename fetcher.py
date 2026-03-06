import arxiv
import fitz
import requests
import json
import os
import time

PAPERS_PER_TOPIC = 8            # number of papers per topic
OUTPUT_FILE = 'papers.json'
PDF_DOWNLOAD_DIR = 'pdfs'       # folder to store downloaded PDFs

# List of AI research topics
TOPICS = [
    "attention mechanism transformer",
    "retrieval augmented generation RAG",
    "LoRA low rank adaptation fine-tuning",
    "diffusion models image generation",
    "reinforcement learning from human feedback RLHF",
    "GPT large language model",
    "BERT pre-training language representation",
    "vision transformer image recognition",
    "chain of thought reasoning LLM",
    "mixture of experts sparse model",
]

# Searches ArXiv and returns paper metadata where each dictionary represents one paper
def search_papers(topic: str, max_results: int) -> list[dict]:
    """Search ArXiv for a topic and return paper metadata."""
    client = arxiv.Client()     # Create a connection
    search = arxiv.Search(
        query=topic,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers = []
    for result in client.results(search):
            # Each paper is stored as a dictionary
            papers.append({
                "title":    result.title,
                "authors":  [str(a) for a in result.authors],
                "year":     result.published.year,
                "abstract": result.summary.replace("\n", " "),
                "url":      result.entry_id,         
                "pdf_url":  result.pdf_url,
                "topic":    topic,
                "text":     "",                        
            })
    return papers

def download_pdf(pdf_url: str, filepath: str) -> bool:
    """Download a PDF from ArXiv. Returns True on success."""
    try:
        response = requests.get(pdf_url, timeout=30)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"      Download failed: {e}")
        return False

# Convert a PDF paper into plain text
def extract_text_from_pdf(filepath: str) -> str:
    """Extract and clean full text from a PDF using PyMuPDF."""
    try:
        doc  = fitz.open(filepath)
        text = ""
        # Extract text page by page
        for page in doc:
            text += page.get_text()
        doc.close()

        # Basic cleanup by removing empty lines / unecessary space
        lines        = [line.strip() for line in text.split("\n") if line.strip()]
        cleaned_text = "\n".join(lines)
        return cleaned_text

    except Exception as e:
        print(f"      Text extraction failed: {e}")
        return ""

def deduplicate(papers: list[dict]) -> list[dict]:
    """Remove duplicate papers defined as same title from multiple topic searches."""
    seen   = set()
    unique = []
    for p in papers:
        key = p["title"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique

def main():
    # Create pdf folder
    os.makedirs(PDF_DOWNLOAD_DIR, exist_ok=True)

    print(" Searching ArXiv...\n")
    all_papers = []

    # Search all Topics
    for topic in TOPICS:
        print(f"   Topic: '{topic}'")
        papers = search_papers(topic, PAPERS_PER_TOPIC)
        all_papers.extend(papers)
        print(f"      -> Found {len(papers)} papers")
        time.sleep(1)  

    all_papers = deduplicate(all_papers)
    print(f"\n Total unique papers found: {len(all_papers)}\n")

    # 2. Download PDFs + extract text 
    print(" Downloading PDFs and extracting text...\n")

    for i, paper in enumerate(all_papers):
        title_short = paper["title"][:60]
        print(f"  [{i+1}/{len(all_papers)}] {title_short}...")

        # Build a safe filename 
        arxiv_id = paper["url"].split("/")[-1]
        pdf_path = os.path.join(PDF_DOWNLOAD_DIR, f"{arxiv_id}.pdf")

        # Skip download if already downloaded
        if os.path.exists(pdf_path):
            print("        PDF already downloaded, skipping")
        else:
            success = download_pdf(paper["pdf_url"], pdf_path)
            if not success:
                continue
            time.sleep(1)  

        # Extract text from the PDF
        text = extract_text_from_pdf(pdf_path)

        if text:
            paper["text"] = text
            print(f"       Extracted {len(text):,} characters")
        else:
            print("       No text extracted — skipping paper")

    # 3. Filter out papers with no text 
    papers_with_text = [p for p in all_papers if p["text"]]
    print(f"\n Papers with usable text: {len(papers_with_text)}/{len(all_papers)}")

    # 4. Save to JSON 
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(papers_with_text, f, ensure_ascii=False, indent=2)

    print(f"\n Done! Saved to '{OUTPUT_FILE}'")
    print(f"   → {len(papers_with_text)} papers available\n")

    # 5. Print a quick summary
    print("=" * 60)
    print("  PAPERS COLLECTED")
    print("=" * 60)
    for p in papers_with_text:
        print(f"  • {p['title'][:55]:<55} ({p['year']})")
    print("=" * 60)


if __name__ == "__main__":
    main()

    