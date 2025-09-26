# scout.py (image-filtered)
"""
scout.py

Improved Scout:
 - Accepts planner output JSON (query + subquestions)
 - For each subquestion: generates keywords (full phrase + tokens)
 - If a keyword is a weak single word (e.g., "what", "recent"), it is NOT used alone:
    - it is attached to a strong token from the original query if available (e.g., GST, India)
    - otherwise attached to the full subquestion text
 - Uses Serper (langchain_community.utilities.GoogleSerperAPIWrapper) for search
 - Uses Firecrawl for content fetch (markdown preferred)
 - Simple relevance filter (token overlap) to drop clearly unrelated pages
 - NEW: removes/filters image-heavy pages and strips image tags from fetched text
"""

import os
import sys
import time
import json
import re
from datetime import datetime
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

# ---------- Configuration / Tunables ----------
TOP_K_DEFAULT = 2
PER_SUBQ_KEYWORDS = 2
SLEEP_BETWEEN_FETCH = 0.5
MIN_TOKEN_LEN = 3
MIN_OVERLAP_RATIO = 0.06  # fraction of core tokens that should appear in fetched doc

# Weak words that should never be searched alone, but must be preserved in context
WEAK_SINGLE_WORDS = {"what", "recent", "latest", "new", "meaning", "definition", "impact", "rules", "how"}

# Domain blacklist (common noisy domains)
DOMAIN_BLACKLIST = {
    "merriam-webster.com", "dictionary.cambridge.org", "wordreference.com",
    "play.google.com", "apps.apple.com", "poki.com", "addictinggames.com",
    "pinimg.com", "instagram.com"
}

# Optional allowlist via env var (comma-separated)
DOMAIN_ALLOWLIST = os.environ.get("SCOUT_DOMAIN_ALLOWLIST")  # e.g., "pib.gov.in,indiatoday.in"

# ---------- Image filtering tunables (NEW) ----------
# Minimum number of non-image words required for pages that contain images.
MIN_TEXT_WORDS_FOR_IMAGES = 40
# If a page has more than this many image markers, it's considered image-heavy (tunable)
IMAGE_MARKER_THRESHOLD = 1

# ---------- External clients (optional) ----------
try:
    from langchain_community.utilities import GoogleSerperAPIWrapper
except Exception as e:
    GoogleSerperAPIWrapper = None
    print("Warning: Serper wrapper not available:", e, file=sys.stderr)

try:
    from firecrawl import Firecrawl
except Exception as e:
    Firecrawl = None
    print("Warning: Firecrawl not available:", e, file=sys.stderr)

SERPER_API_KEY = os.environ.get("SERPER_API_KEY") or os.environ.get("GOOGLE_SERPER_API_KEY")
FIRECRAWL_API_KEY = os.environ.get("FIRECRAWL_API_KEY") or os.environ.get("FC_API_KEY")

serper = None
firecrawl = None
if GoogleSerperAPIWrapper and SERPER_API_KEY:
    try:
        os.environ["SERPER_API_KEY"] = SERPER_API_KEY
        serper = GoogleSerperAPIWrapper()
    except Exception as e:
        print("Warning: Failed to init Serper:", e, file=sys.stderr)
if Firecrawl and FIRECRAWL_API_KEY:
    try:
        firecrawl = Firecrawl(api_key=FIRECRAWL_API_KEY)
    except Exception as e:
        print("Warning: Failed to init Firecrawl:", e, file=sys.stderr)

# ---------- Helpers ----------
def is_valid_url(u):
    try:
        p = urlparse(u)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except:
        return False

def domain_of(url):
    try:
        return urlparse(url).netloc.lower()
    except:
        return ""

def is_domain_allowed(url):
    dom = domain_of(url)
    if not dom:
        return False
    if DOMAIN_ALLOWLIST:
        allow = [d.strip().lower() for d in DOMAIN_ALLOWLIST.split(",") if d.strip()]
        return any(a in dom for a in allow)
    for bad in DOMAIN_BLACKLIST:
        if bad in dom:
            return False
    return True

def token_list_from_text(text):
    toks = [t for t in re.findall(r"[A-Za-z0-9\-]+", (text or ""))]
    filtered = [t.lower() for t in toks if len(t) >= MIN_TOKEN_LEN]
    seen = set()
    out = []
    for t in filtered:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out

# ---------- Image detection & cleanup helpers (NEW) ----------
IMG_MD_RE = re.compile(r'!\[.?\]\((.?)\)')           # markdown images ![alt](url)
IMG_HTML_RE = re.compile(r'<img\b[^>]*>', flags=re.I)  # html <img ...>
IMG_URL_RE = re.compile(r'https?://[^\s)]+?\.(?:png|jpe?g|gif|svg)(?:\?[^\s)]*)?', flags=re.I)

def count_image_markers(text):
    """Return count of markdown/html/image-url occurrences in text."""
    if not text:
        return 0
    md = len(IMG_MD_RE.findall(text))
    html = len(IMG_HTML_RE.findall(text))
    url_imgs = len(IMG_URL_RE.findall(text))
    return max(md, html, url_imgs)  # use max to avoid double-counting same image in different forms

def strip_image_tags(text):
    """Remove markdown image tags, HTML <img> tags, and standalone image URLs."""
    if not text:
        return text
    text = IMG_MD_RE.sub('', text)
    text = IMG_HTML_RE.sub('', text)
    # remove standalone image URLs (common in scraped markdown)
    text = IMG_URL_RE.sub('', text)
    return text

# ---------- Search wrapper ----------
def search_serper(keyword, top_k=TOP_K_DEFAULT, recency_days=None):
    urls = []
    if serper is None:
        print("[SERP] Serper wrapper not initialized; returning empty URL list", file=sys.stderr)
        return urls
    print(f"[SERP] Searching for: {keyword} (top_k={top_k}, recency_days={recency_days})", file=sys.stderr)
    try:
        result_json = None
        if recency_days:
            try:
                result_json = serper.results(keyword, params={"time_period_days": recency_days})
            except TypeError:
                try:
                    result_json = serper.results(keyword, time_period=recency_days)
                except TypeError:
                    result_json = serper.results(keyword)
        else:
            result_json = serper.results(keyword)

        if isinstance(result_json, dict) and 'organic' in result_json:
            for organic in result_json['organic']:
                link = organic.get('link') or organic.get('url') or organic.get('source')
                if link and is_valid_url(link):
                    urls.append(link)
                    if len(urls) >= top_k:
                        break
        elif isinstance(result_json, dict) and 'items' in result_json:
            for item in result_json['items']:
                link = item.get('link') or item.get('url')
                if link and is_valid_url(link):
                    urls.append(link)
                    if len(urls) >= top_k:
                        break
        else:
            txt = json.dumps(result_json) if result_json is not None else ""
            found = re.findall(r'https?://[^\s,"\\\']+', txt)
            for l in found:
                if is_valid_url(l):
                    urls.append(l)
                    if len(urls) >= top_k:
                        break
    except Exception as e:
        print(f"[SERP] Error searching '{keyword}': {e}", file=sys.stderr)
    print(f"[SERP] URLs: {urls}", file=sys.stderr)
    return urls

# ---------- Fetch wrapper (updated to strip images and skip image-heavy pages) ----------
def fetch_with_firecrawl(url):
    """
    Returns: (title, cleaned_text, pub_date)
    - cleaned_text has image tags/URLs removed.
    - If the page is image-heavy and contains too little text, returns ("", "", None) to indicate skip.
    """
    if firecrawl is None:
        print("[FETCH] Firecrawl not initialized; cannot fetch", file=sys.stderr)
        return "", "", None
    print(f"[FETCH] Fetching: {url}", file=sys.stderr)
    try:
        resp = firecrawl.scrape(url, formats=["markdown"])
        markdown_text = getattr(resp, "markdown", None) or getattr(resp, "text", "") or ""
        meta = getattr(resp, "meta", None)
        pub_date = None
        if isinstance(meta, dict):
            for k in ("date", "published", "timestamp", "pub_date"):
                if k in meta:
                    pub_date = meta[k]
                    break
        title = ""
        if markdown_text:
            for line in markdown_text.splitlines():
                if line.startswith("# "):
                    title = line[2:].strip()
                    break

        # Image analysis + cleanup
        img_count = count_image_markers(markdown_text)
        # cleaned text with image tags/URLs removed
        cleaned = strip_image_tags(markdown_text)
        # count word tokens in cleaned text
        word_count = len(re.findall(r'\w+', cleaned or ""))

        # if page is image-heavy and has low textual content, skip it
        if img_count >= IMAGE_MARKER_THRESHOLD and word_count < MIN_TEXT_WORDS_FOR_IMAGES:
            print(f"[FETCH] Skipping {url} — image_count={img_count}, word_count={word_count} (image-heavy)", file=sys.stderr)
            return "", "", None

        # otherwise return cleaned text (images removed)
        return title, cleaned, pub_date

    except Exception as e:
        print(f"[FETCH] Error fetching {url}: {e}", file=sys.stderr)
        return "", "", None

def text_relevance_score(text, core_tokens):
    if not core_tokens:
        return 0.0
    txt = (text or "").lower()
    count = 0
    for t in set(core_tokens):
        if t.lower() in txt:
            count += 1
    return count / float(len(core_tokens))

# ---------- KEYWORD generation + weak-word handling ----------
def generate_candidates(stext, original_query_tokens, per_subq=PER_SUBQ_KEYWORDS):
    candidates = []
    if stext:
        phrase = re.sub(r'\?+$', '', stext).strip()
        if phrase:
            candidates.append(phrase)

    toks = [t for t in re.findall(r"[A-Za-z0-9\-]+", stext)]
    for t in toks:
        if t not in candidates:
            candidates.append(t)

    if not candidates:
        candidates = [stext or "query"]

    candidates = candidates[:per_subq]

    processed = []
    for kw in candidates:
        words = re.findall(r"[A-Za-z0-9\-]+", kw)
        if len(words) == 1 and words[0].lower() in WEAK_SINGLE_WORDS:
            strong = None
            for tok in original_query_tokens:
                if len(tok) >= MIN_TOKEN_LEN and tok.lower() not in WEAK_SINGLE_WORDS:
                    strong = tok
                    break
            if strong:
                new_kw = f"{words[0]} {strong}"
            else:
                new_kw = f"{words[0]} {stext}".strip()
            if new_kw not in processed:
                processed.append(new_kw)
        else:
            if kw not in processed:
                processed.append(kw)
    return processed[:per_subq]

# ---------- Main scout function ----------
def scout(subqs, top_k=TOP_K_DEFAULT, per_subq_keywords=PER_SUBQ_KEYWORDS,
          sleep_between_fetch=SLEEP_BETWEEN_FETCH, original_query="", recency_days=None):
    print(f"[SCOUT] Starting scout for {len(subqs)} subquestions", file=sys.stderr)
    results = []
    original_tokens = token_list_from_text(original_query) if original_query else []

    for sq in subqs:
        sid = sq.get("id")
        stext = sq.get("text") or ""
        candidates = generate_candidates(stext, original_tokens, per_subq=per_subq_keywords)
        print(f"[SCOUT] SubQ {sid} -> candidates: {candidates}", file=sys.stderr)

        core_tokens = list(dict.fromkeys(original_tokens + token_list_from_text(stext)))

        for kw in candidates:
            urls = search_serper(kw, top_k=top_k, recency_days=recency_days)
            for url in urls:
                if not is_domain_allowed(url):
                    print(f"[SCOUT] Skipping blacklisted domain: {url}", file=sys.stderr)
                    continue
                title, text, pub_date = fetch_with_firecrawl(url)
                if not (title or text):
                    # skipped due to image-heavy or fetch error
                    continue
                if recency_days and pub_date:
                    try:
                        parsed = None
                        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%d %b %Y", "%b %d, %Y"):
                            try:
                                parsed = datetime.strptime(pub_date, fmt)
                                break
                            except Exception:
                                parsed = None
                        if parsed:
                            age_days = (datetime.utcnow() - parsed).days
                            if age_days > recency_days:
                                print(f"[SCOUT] Discarding {url} due to publish age {age_days}d > {recency_days}d", file=sys.stderr)
                                continue
                    except Exception:
                        pass

                score = text_relevance_score((title or "") + "\n" + (text or ""), core_tokens)
                print(f"[SCOUT] Relevance score for {url}: {score:.3f}", file=sys.stderr)
                if score < MIN_OVERLAP_RATIO:
                    print(f"[SCOUT] Discarding {url} due to low overlap", file=sys.stderr)
                    continue

                results.append({
                    "subq_id": sid,
                    "subq_text": stext,
                    "keyword": kw,
                    "url": url,
                    "title": title,
                    "text": text,
                    "pub_date": pub_date,
                    "relevance_score": score,
                    "fetched_at": datetime.utcnow().isoformat() + "Z"
                })
                time.sleep(sleep_between_fetch)

    print(f"[SCOUT] Completed. Relevant results fetched: {len(results)}", file=sys.stderr)
    return results

# ---------- CLI ----------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Scout: search & fetch for planner subquestions (weak-word-safe, image-filtered)")
    parser.add_argument("--input", default="-", help="Path to planner JSON (or '-' to read stdin)")
    parser.add_argument("--top_k", type=int, default=TOP_K_DEFAULT, help="Top URLs per keyword")
    parser.add_argument("--per_kw", type=int, default=PER_SUBQ_KEYWORDS, help="Keywords per subquestion")
    parser.add_argument("--recency_days", type=int, default=None, help="Prefer results within this many days (optional)")
    args = parser.parse_args()

    raw = ""
    if args.input == "-":
        raw = sys.stdin.read()
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            raw = f.read()

    try:
        plan = json.loads(raw)
    except Exception:
        try:
            plan = {"subquestions": json.loads(raw)}
        except Exception as e:
            print("Failed to parse input JSON:", e, file=sys.stderr)
            raise SystemExit(1)

    subqs = plan.get("subquestions") or []
    original_query = plan.get("query", "")
    results = scout(subqs, top_k=args.top_k, per_subq_keywords=args.per_kw,
                    sleep_between_fetch=SLEEP_BETWEEN_FETCH, original_query=original_query,
                    recency_days=args.recency_days)
    print(json.dumps(results, indent=2, ensure_ascii=False))
