import os
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INDEX_NAME = "company-policies"
PDF_PATH   = "Company Policy Document.pdf"

# ─────────────────────────────────────────────
# INIT PINECONE
# ─────────────────────────────────────────────
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Create index if not exists
if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

index = pc.Index(INDEX_NAME)

# ─────────────────────────────────────────────
# EMBEDDING (shared everywhere)
# ─────────────────────────────────────────────
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ─────────────────────────────────────────────
# VECTORSTORE (READ-ONLY)
# ─────────────────────────────────────────────
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embedding
)

# ─────────────────────────────────────────────
# INGESTION (ONLY WHEN NEEDED)
# ─────────────────────────────────────────────
def _ingest_policies():
    print("⚠️ Ingesting company policies into Pinecone...")

    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_documents(documents)

    PineconeVectorStore.from_documents(
        docs,
        embedding,
        index_name=INDEX_NAME
    )

    print("✅ Policy data ingested successfully.")


# ─────────────────────────────────────────────
# AUTO-CHECK (SMART LOADER)
# ─────────────────────────────────────────────
def ensure_policy_data_loaded():
    stats = index.describe_index_stats()
    total_vectors = stats.get("total_vector_count", 0)

    if total_vectors > 0:
        print("✅ Policy data already present. Skipping ingestion.")
        return

    _ingest_policies()