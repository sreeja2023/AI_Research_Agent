"""
Step 2: Scout — Search & Scrape
"""

import time, requests
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import urlparse
from duckduckgo_search import DDGS

def is_valid_url(u):
    try:
        p = urlparse(u)
        return p.scheme in ("http","https") and bool(p.netloc)
    except:
        return False

def search_duckduckgo(keyword, top_k=2):
    urls = []
    with DDGS() as ddgs:
        results = ddgs.text(keyword, max_results=top_k)
        for r in results:
            url = r.get("href") or r.get("url")
            if url and is_valid_url(url):
                urls.append(url)
    return urls

def fetch_text(url):
    try:
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        title = soup.title.string.strip() if soup.title else ""
        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all(["p","li"])]
        return title, "\n".join(paragraphs)
    except:
        return "", ""

def scout(subqs, top_k=2):
    results = []
    for sq in subqs:
        sid, stext = sq["id"], sq["text"]
        keywords = sq.get("search_keywords") or [stext]
        for kw in keywords[:2]:
            urls = search_duckduckgo(kw, top_k=top_k)
            for url in urls:
                title, text = fetch_text(url)
                results.append({
                    "subq_id": sid,
                    "subq_text": stext,
                    "keyword": kw,
                    "url": url,
                    "title": title,
                    "text": text,
                    "fetched_at": datetime.utcnow().isoformat()+"Z"
                })
                time.sleep(0.5)
    return results

if __name__ == "__main__":
    # Dummy test
    demo_subs = [{"id":"Q1","text":"H1B visa rules 2025","search_keywords":["H1B rules 2025"]}]
    print(scout(demo_subs))
