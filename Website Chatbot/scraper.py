import asyncio
import re
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

ROUTES = [
    "/", "/about", "/services", "/Industries",
    "/technology", "/project", "/blog", "/contact" , "/career" , '/blog/cybersecurity-best-practices-how-to-stay-safe-online-2024-fastcadcoding' ,
    '/blog/what-is-react-and-how-to-learn-it-fastcadcoding' , '/policy'
]

async def crawl_full_site(base_url: str) -> list[dict]:
    results = []

    browser_config = BrowserConfig(headless=True, verbose=False)
    run_config = CrawlerRunConfig(
        wait_until="networkidle",
        delay_before_return_html=6.0,
        page_timeout=60000,
        word_count_threshold=5,
        js_code="window.scrollTo(0, document.body.scrollHeight);",
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for route in ROUTES:
            url = base_url.rstrip("/") + route
            print(f"\n{'='*60}")
            print(f"Crawling: {url}")
            print(f"{'='*60}")

            result = await crawler.arun(url=url, config=run_config)

            # Get content from markdown, fallback to stripped HTML
            content = ""
            if result.markdown:
                content = result.markdown.raw_markdown or ""
            if not content and result.cleaned_html:
                content = re.sub(r'<[^>]+>', ' ', result.cleaned_html)
                content = re.sub(r'\s+', ' ', content).strip()

            if content and len(content) > 50:
                results.append({"url": url, "content": content})
                print(f"✅ URL     : {url}")
                print(f"📄 Content Preview (first 500 chars):\n")
                print(content)
                print(f"\n... [{len(content)} total characters]")
            else:
                print(f"❌ No content for: {url}")

    print(f"\n{'='*60}")
    print(f"✅ Crawl Complete — Total pages scraped: {len(results)}")
    print(f"{'='*60}\n")
    return results


async def crawl_single_page(url: str) -> dict | None:
    browser_config = BrowserConfig(headless=True, verbose=False)
    run_config = CrawlerRunConfig(
        wait_until="networkidle",
        delay_before_return_html=6.0,
        page_timeout=60000,
        word_count_threshold=5,
        js_code="window.scrollTo(0, document.body.scrollHeight);",
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        print(f"\nCrawling single page: {url}")
        result = await crawler.arun(url=url, config=run_config)

        content = ""
        if result.markdown:
            content = result.markdown.raw_markdown or ""
        if not content and result.cleaned_html:
            content = re.sub(r'<[^>]+>', ' ', result.cleaned_html)
            content = re.sub(r'\s+', ' ', content).strip()

        if content and len(content) > 50:
            print(f"✅ {url} — {len(content)} chars")
            print(content[:300])
            return {"url": url, "content": content}

        print(f"❌ No content for: {url}")
        return None


if __name__ == "__main__":
    asyncio.run(crawl_full_site("https://www.detagenix.com"))