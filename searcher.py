"""
searcher.py — web search for THREAD
Pulls real articles per topic + per interrogation round using DuckDuckGo.
"""

import re
import time
import requests
from ddgs import DDGS

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}

# ── Blocked domains ───────────────────────────────────────────────────────────
BLOCKED_DOMAINS = {
    # non-English
    "baidu.com", "zhidao.baidu.com", "zhihu.com", "weibo.com",
    "sina.com.cn", "163.com", "sohu.com", "qq.com",
    # social / video
    "reddit.com", "pinterest.com", "instagram.com",
    "twitter.com", "x.com", "tiktok.com", "facebook.com", "youtube.com",
    # shopping
    "amazon.com", "ebay.com", "etsy.com",
    # news aggregators
    "msn.com", "yahoo.com", "news.google.com", "flipboard.com",
    "smartnews.com", "feedly.com",
    # JS-walled
    "economist.com",
    # tech / gaming / entertainment — off-topic for serious queries
    "stackoverflow.com", "stackexchange.com", "superuser.com",
    "github.com", "medium.com", "substack.com",
    "ign.com", "nordic.ign.com", "kotaku.com", "polygon.com",
    "listennotes.com", "spotify.com", "podcasts.apple.com",
    "buzzfeed.com", "vice.com", "vox.com",
    "quora.com", "answers.yahoo.com",
    # low-quality blogs / personal sites caught by pattern
    "worldpeacefull.com", "wpas.worldpeacefull.com",
}

# ── Preferred domains — authoritative sources ──────────────────────────────────
PREFERRED_DOMAINS = {
    # global news
    "bbc.com", "bbc.co.uk", "reuters.com", "apnews.com",
    "theguardian.com", "nytimes.com", "washingtonpost.com",
    "aljazeera.com", "dw.com", "france24.com", "time.com",
    "theatlantic.com",
    # foreign policy / IR
    "foreignaffairs.com", "foreignpolicy.com",
    "nationalinterest.org", "cfr.org", "chathamhouse.org",
    "lowyinstitute.org", "carnegieendowment.org",
    # academic / think tank
    "brookings.edu", "rand.org", "jstor.org", "nature.com",
    "sciencedirect.com", "wikipedia.org", "britannica.com",
    "plato.stanford.edu", "iep.utm.edu",
    "nyupress.org", "researchgate.net", "academia.edu",
    # historical
    "history.com", "historyextra.com", "thehistorypress.co.uk",
    # open-access academic
    "archive.org", "books.google.com",
    "plato.stanford.edu", "iep.utm.edu",
    "semanticscholar.org", "philpapers.org",
    # policy/IR
    "fpri.org", "chathamhouse.org", "crisisgroup.org",
    "usip.org", "wilsoncenter.org",
}

# ── Snippet quality filters ────────────────────────────────────────────────────
# Reject articles whose snippet looks like JS walls or pure navigation
JUNK_SNIPPET_SIGNALS = [
    "enable javascript", "just a moment", "please enable",
    "subscribe to continue", "sign in to read",
    "cookies to continue", "403 forbidden",
    "access denied", "temporarily unavailable", "ray id",
    "you do not have access", "cloudflare", "checking your browser",
    "cf-ray", "please verify", "basket", "skip to navigation",
    "skip to content", "mailing list", "browse by subject",
    # concatenated titles with no real content
    "taylor & francis", "subscribe subscribe login",
    "ad feedback", "cookie policy", "privacy policy",
    "purchase a subscription", "to continue reading",
    "subscribe to read", "register to read", "create a free account",
    "already a subscriber", "get full access", "unlock this article",
]

# Domains that consistently block scrapers
SCRAPER_BLOCKED = {
    "researchgate.net", "academia.edu", "jstor.org",
    "sciencedirect.com", "springer.com", "tandfonline.com",
    "historytoday.com", "foreignaffairs.com", "newyorker.com",
    "theatlantic.com", "wired.com", "thetimes.co.uk",
    "ft.com", "wsj.com", "bloomberg.com", "hbr.org",
    # news sites that return off-topic breaking news for general queries
    "cnn.com", "foxnews.com", "nbcnews.com", "abcnews.go.com",
    "cbsnews.com", "msnbc.com", "nypost.com", "dailymail.co.uk",
}

# ── Per-round search queries — specific, not generic ─────────────────────────
# Each round runs MULTIPLE targeted queries to get diverse sources
def build_queries(topic: str, round_type: str | None) -> list[str]:
    """
    Build search queries for a topic + round. Avoids putting the raw topic
    in quotes (breaks DDG) and keeps queries specific enough to avoid garbage.
    """
    t = topic  # e.g. "why wars happen"
    # strip leading "why/how/what" for cleaner sub-queries
    import re as _re
    core = _re.sub(r"^(why|how|what|when|where|who)\s+", "", t, flags=_re.IGNORECASE).strip()
    # e.g. "wars happen" → use as anchor

    templates = {
        None: [
            f"{core} causes theories historians",
            f"{core} political science explanation",
            f"{core} historical analysis academic",
        ],
        "mechanism": [
            f"{core} causal mechanism how",
            f"{core} root causes political science",
            f"why do {core} scholarly explanation",
        ],
        "counterexample": [
            f"{core} counterexample historical case failed",
            f"peace instead of {core} examples",
            f"{core} disproved exception evidence",
        ],
        "evidence": [
            f"{core} empirical evidence academic study",
            f"{core} research data findings statistics",
            f"academic paper {core} analysis",
        ],
        "consistency": [
            f"{core} debate competing theories scholars",
            f"criticism of {core} theory alternative",
            f"{core} contradictions disagreement scholars",
        ],
    }
    return templates.get(round_type, templates[None])

# kept for API compatibility
ROUND_QUERIES = {}


def _domain(url: str) -> str:
    m = re.match(r"https?://(?:www\.)?([^/]+)", url or "")
    return m.group(1) if m else ""


# ── Source tier + type classification ────────────────────────────────────────
# Tier 1: peer-reviewed, primary data, government stats
# Tier 2: reputable journalism, think tanks, established reference
# Tier 3: blog, opinion, unknown

TIER1_DOMAINS = {
    "pubmed.ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov", "pnas.org",
    "nature.com", "science.org", "cell.com", "thelancet.com",
    "nejm.org", "bmj.com", "jamanetwork.com",
    "scholar.google.com", "semanticscholar.org",
    "jstor.org", "cambridge.org", "oup.com",
    "rand.org", "brookings.edu", "who.int", "un.org",
    "data.worldbank.org", "ourworldindata.org",
    "web.stanford.edu", "mit.edu", "harvard.edu",
    "pmc.ncbi.nlm.nih.gov", "journals.plos.org",
    "academic.oup.com", "onlinelibrary.wiley.com",
    "ssrn.com", "arxiv.org", "medrxiv.org",
    "ppublishing.org", "cambridge.org", "pnas.org",
    "academiccommons.columbia.edu", "journals.flvc.org",
}

TIER1_SOURCE_TYPES = {
    "pubmed.ncbi.nlm.nih.gov": "peer-reviewed",
    "pmc.ncbi.nlm.nih.gov":    "peer-reviewed",
    "pnas.org":                "peer-reviewed",
    "nature.com":              "peer-reviewed",
    "science.org":             "peer-reviewed",
    "ourworldindata.org":      "primary-data",
    "rand.org":                "think-tank",
    "brookings.edu":           "think-tank",
    "who.int":                 "government-data",
    "arxiv.org":               "preprint",
    "ssrn.com":                "preprint",
}

TIER2_DOMAINS = {
    "bbc.com", "bbc.co.uk", "reuters.com", "apnews.com",
    "theguardian.com", "nytimes.com", "washingtonpost.com",
    "aljazeera.com", "dw.com", "france24.com", "time.com",
    "foreignaffairs.com", "foreignpolicy.com",
    "nationalinterest.org", "cfr.org", "chathamhouse.org",
    "fpri.org", "crisisgroup.org", "usip.org", "wilsoncenter.org",
    "britannica.com", "wikipedia.org", "historyextra.com",
    "smithsonianmag.com", "scientificamerican.com",
    "theatlantic.com", "press.uchicago.edu", "nyupress.org",
    "palladiummag.com", "scispace.com", "resilience.org",
    "archive.org", "books.google.com",
    "mwi.westpoint.edu", "activehistory.co.uk",
    "polsci.institute", "historytoday.com",
}


def get_source_tier(url: str) -> tuple[int, str]:
    """Return (tier, source_type) for a URL."""
    d = _domain(url)
    if any(d == t or d.endswith("." + t) for t in TIER1_DOMAINS):  # noqa
        stype = TIER1_SOURCE_TYPES.get(d, "peer-reviewed")
        return 1, stype
    if any(d == t or d.endswith("." + t) for t in TIER2_DOMAINS):
        # journalism vs think-tank vs reference
        if any(x in d for x in ["rand","brookings","cfr","fpri","chatham","crisisgroup","usip","wilson","palladium"]):
            return 2, "think-tank"
        if any(x in d for x in ["britannica","wikipedia","archive","books.google"]):
            return 2, "expert-opinion"
        return 2, "journalism"
    return 3, "blog"


def _is_blocked(url: str) -> bool:
    d = _domain(url)
    return (
        any(d == b or d.endswith("." + b) for b in BLOCKED_DOMAINS) or
        any(d == b or d.endswith("." + b) for b in SCRAPER_BLOCKED)
    )


def _is_preferred(url: str) -> bool:
    d = _domain(url)
    return any(d == p or d.endswith("." + p) for p in PREFERRED_DOMAINS)


def _is_junk_snippet(snippet: str) -> bool:
    """Reject JS walls, paywalls, empty pages."""
    s = snippet.lower()
    return any(sig in s for sig in JUNK_SNIPPET_SIGNALS) or len(snippet.strip()) < 40


def _score(article: dict) -> int:
    """Higher = better. Used to sort merged results."""
    score = 0
    if _is_preferred(article["url"]):
        score += 10
    if article.get("published"):
        score += 2
    snip = article.get("snippet", "")
    if len(snip) > 200:
        score += 2
    elif len(snip) > 100:
        score += 1
    return score


def search_web(query: str, max_results: int = 6) -> list[dict]:
    results = []
    try:
        with DDGS() as ddg:
            for r in ddg.text(query, max_results=max_results * 3):  # fetch extra, filter down
                url     = r.get("href", "")
                snippet = r.get("body", "")
                if not url or _is_blocked(url):
                    continue
                if _is_junk_snippet(snippet):
                    continue
                tier, stype = get_source_tier(url)
                results.append({
                    "title":       r.get("title", ""),
                    "url":         url,
                    "snippet":     snippet,
                    "source":      _domain(url),
                    "published":   "",
                    "tier":        tier,
                    "source_type": stype,
                })
                if len(results) >= max_results:
                    break
        time.sleep(0.4)
    except Exception as e:
        print(f"  [searcher] web search failed: {e}")
    return results


def search_news(query: str, max_results: int = 5) -> list[dict]:
    results = []
    try:
        with DDGS() as ddg:
            for r in ddg.news(query, max_results=max_results * 2):
                url = r.get("url", "")
                if not url or _is_blocked(url):
                    continue
                tier, stype = get_source_tier(url)
                results.append({
                    "title":       r.get("title", ""),
                    "url":         url,
                    "snippet":     r.get("body", ""),
                    "source":      r.get("source", _domain(url)),
                    "published":   r.get("date", ""),
                    "tier":        tier,
                    "source_type": stype,
                })
                if len(results) >= max_results:
                    break
        time.sleep(0.4)
    except Exception as e:
        print(f"  [searcher] news search failed: {e}")
    return results


def fetch_article(url: str, max_chars: int = 1800) -> str:
    try:
        r = requests.get(url, headers=HEADERS, timeout=8, allow_redirects=True)
        if "text/html" not in r.headers.get("Content-Type", ""):
            return ""
        text = re.sub(r"<script[^>]*>.*?</script>", "", r.text, flags=re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>",   "", text,    flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except Exception:
        return ""


def build_sources_for_round(topic: str, round_type: str | None,
                             max_articles: int = 8, verbose: bool = True,
                             seen_urls: set = None) -> list[dict]:
    """
    Run multiple targeted queries for this round, merge + deduplicate,
    filter blocked domains, prefer authoritative sources.
    seen_urls: urls already fetched in previous rounds (avoids repeating same article)
    """
    if seen_urls is None:
        seen_urls = set()

    label   = round_type or "initial"
    queries = build_queries(topic, round_type)

    if verbose:
        print(f"  [searcher] Round [{label}] — {len(queries)} queries:")

    all_articles = []
    seen_this_round = set()

    for q in queries:
        if verbose:
            print(f"             → \"{q}\"")

        web  = search_web(q,     max_results=5)
        news = search_news(topic, max_results=3)

        for a in web + news:
            url = a["url"]
            if not url:
                continue
            if url in seen_this_round:
                continue
            if url in seen_urls:
                continue  # skip articles used in a previous round
            if _is_blocked(url):
                continue
            seen_this_round.add(url)
            all_articles.append(a)

        time.sleep(0.3)

    # sort: preferred domains first, then by snippet richness
    all_articles.sort(key=_score, reverse=True)

    # enrich top 3 with full article text
    for a in all_articles[:3]:
        full = fetch_article(a["url"])
        if full:
            a["snippet"] = full

    result = all_articles[:max_articles]

    if verbose:
        domains = [_domain(a["url"]) for a in result]
        print(f"  [searcher] {len(result)} sources: {domains}\n")

    return result


def format_sources_for_prompt(articles: list[dict]) -> str:
    if not articles:
        return "(No external sources — reasoning from model knowledge only)"

    lines = [
        "SOURCES — use these to ground your reasoning. Cite inline as [1], [2] etc.\n",
        "Do NOT let a single source dominate. Synthesize across sources.\n",
    ]
    for i, a in enumerate(articles, 1):
        lines.append(f"[{i}] {a['title']}")
        lines.append(f"    {a['source']}" + (f"  ·  {a['published'][:10]}" if a.get("published") else ""))
        if a.get("snippet"):
            lines.append(f"    {a['snippet'][:350]}")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    topic = " ".join(sys.argv[1:]) or "why wars happen"
    arts  = build_sources_for_round(topic, None, verbose=True)
    print(format_sources_for_prompt(arts))