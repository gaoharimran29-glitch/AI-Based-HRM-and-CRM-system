import os
import hashlib
import fitz  # pymupdf
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ── Model & Pinecone init ─────────────────────────────────────────────────────

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")  # fast, free, good quality
EMBEDDING_DIM = 384  # dimension for all-MiniLM-L6-v2

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

# ── Pinecone index ────────────────────────────────────────────────────────────

def get_or_create_index():
    existing = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing:
        print(f"Creating Pinecone index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(INDEX_NAME)

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_id(source: str, chunk_index: int) -> str:
    base = hashlib.md5(source.encode()).hexdigest()
    return f"{base}_{chunk_index}"

def embed_texts(texts: list[str]) -> list[list[float]]:
    return model.encode(texts, show_progress_bar=True).tolist()

def upsert_chunks(index, chunks: list[str], source: str, source_type: str):
    """Embed and upsert chunks into Pinecone with metadata."""
    if not chunks:
        return

    print(f"  Embedding {len(chunks)} chunks from: {source}")
    vectors = embed_texts(chunks)

    pinecone_vectors = [
        {
            "id": make_id(source, i),
            "values": vector,
            "metadata": {
                "text": chunk,
                "source": source,
                "type": source_type  # "website" or "pdf"
            }
        }
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(pinecone_vectors), batch_size):
        index.upsert(vectors=pinecone_vectors[i:i + batch_size])

    print(f"  ✅ Upserted {len(pinecone_vectors)} vectors for: {source}")

# ── Website pages ─────────────────────────────────────────────────────────────

def upsert_page(page: dict):
    """
    Takes a {url, content} dict from scraper.py and stores it in Pinecone.
    Deletes old vectors for this URL first so updates are clean.
    """
    index = get_or_create_index()
    url = page["url"]
    content = page["content"]

    # Delete old vectors for this URL
    try:
        index.delete(filter={"source": url})
    except Exception:
        pass

    chunks = text_splitter.split_text(content)
    upsert_chunks(index, chunks, source=url, source_type="website")


def upsert_all_pages(pages: list[dict]):
    """Upsert all scraped website pages."""
    print(f"\n{'='*60}")
    print(f"Upserting {len(pages)} website pages into Pinecone...")
    print(f"{'='*60}")
    for page in pages:
        upsert_page(page)
    print("\n✅ All website pages upserted.\n")

# ── PDF files ─────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text


def upsert_pdf(pdf_path: str):
    """
    Reads a PDF, chunks it, embeds it, and upserts into Pinecone.
    source will be the filename so it's identifiable in metadata.
    """
    index = get_or_create_index()
    filename = os.path.basename(pdf_path)

    print(f"\nProcessing PDF: {filename}")

    # Delete old vectors for this PDF if re-uploading
    try:
        index.delete(filter={"source": filename})
    except Exception:
        pass

    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        print(f"  ❌ No text found in {filename}")
        return

    print(f"  Extracted {len(text)} characters from PDF")
    chunks = text_splitter.split_text(text)
    upsert_chunks(index, chunks, source=filename, source_type="pdf")


def upsert_all_pdfs(pdf_folder: str):
    """
    Pass a folder path — it will find and upsert all PDFs inside it.
    Example: upsert_all_pdfs("./pdfs")
    """
    print(f"\n{'='*60}")
    print(f"Looking for PDFs in: {pdf_folder}")
    print(f"{'='*60}")

    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    if not pdf_files:
        print("  ❌ No PDF files found.")
        return

    for filename in pdf_files:
        full_path = os.path.join(pdf_folder, filename)
        upsert_pdf(full_path)

    print("\n✅ All PDFs upserted.\n")

# ── Search ────────────────────────────────────────────────────────────────────

def search(query: str, top_k: int = 5, source_type: str = None) -> list[dict]:
    """
    Search Pinecone for relevant chunks.
    Optionally filter by source_type: 'website' or 'pdf'
    """
    index = get_or_create_index()
    query_vector = model.encode(query).tolist()

    filter_dict = {}
    if source_type:
        filter_dict["type"] = source_type

    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter=filter_dict if filter_dict else None
    )

    return [
        {
            "text": match.metadata["text"],
            "source": match.metadata["source"],
            "type": match.metadata["type"],
            "score": match.score
        }
        for match in results.matches
    ]