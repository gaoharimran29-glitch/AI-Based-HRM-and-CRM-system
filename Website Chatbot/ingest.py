import asyncio
from scraper import crawl_full_site
from vector_store import upsert_all_pages, upsert_all_pdfs, get_or_create_index

async def main():
    index = get_or_create_index()

    # ── Wipe everything before re-ingesting ──────────────────────────────────
    print("Deleting all existing vectors from Pinecone...")
    index.delete(delete_all=True)
    print("✅ Pinecone index cleared\n")

    # ── Re-scrape website ─────────────────────────────────────────────────────
    print("Starting website crawl...")
    pages = await crawl_full_site("https://www.detagenix.com")
    upsert_all_pages(pages)

    # ── Re-upsert PDFs ────────────────────────────────────────────────────────
    upsert_all_pdfs("./pdfs")

    print("\n🎉 All data re-ingested into Pinecone successfully!")

asyncio.run(main())