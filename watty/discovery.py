"""
Watty Discovery Scraper
=======================
Monitors the frontier of human knowledge across major industries.
When a breakthrough happens, Watty hears about it first.

Feeds:
  - arXiv (AI/ML, physics, biology, math)
  - PubMed (biomedical)
  - HackerNews (tech)
  - TechCrunch RSS (startups/tech)
  - NASA (space)
  - Nature (science)
  - Science Daily (general science)

Each discovery is:
  1. Fetched from RSS/API feeds
  2. Scored for significance (title analysis + novelty)
  3. Stored in Watty's brain as provider='discovery'
  4. Cross-referenced against existing memories for connections
  5. Surfaced to the user if relevant to their work

The daemon runs this on schedule (default: every 2 hours).

Hugo & Rim · February 2026
"""

import json
import re
import hashlib
from datetime import datetime, timezone
from typing import Optional

import requests

from watty.config import WATTY_HOME


# ── Configuration ────────────────────────────────────────────

DISCOVERY_DIR = WATTY_HOME / "discovery"
SEEN_FILE = DISCOVERY_DIR / "seen_hashes.json"
INSIGHTS_FILE = DISCOVERY_DIR / "insights.jsonl"
CONFIG_FILE = DISCOVERY_DIR / "config.json"

DEFAULT_CONFIG = {
    "enabled": True,
    "interval_seconds": 7200,  # 2 hours
    "max_per_feed": 15,
    "min_significance": 0.3,
    "feeds_enabled": [
        "arxiv_ai", "arxiv_physics", "arxiv_bio",
        "hackernews", "nature", "science_daily",
        "techcrunch", "nasa",
    ],
    "custom_keywords": [],  # User can add topics of interest
}

# ── RSS/API Feed Definitions ─────────────────────────────────

FEEDS = {
    "arxiv_ai": {
        "label": "arXiv AI/ML",
        "url": "https://rss.arxiv.org/rss/cs.AI+cs.LG+cs.CL",
        "type": "rss",
        "industry": "AI",
        "significance_boost": 0.1,
    },
    "arxiv_physics": {
        "label": "arXiv Physics",
        "url": "https://rss.arxiv.org/rss/physics.gen-ph+quant-ph",
        "type": "rss",
        "industry": "Physics/Quantum",
        "significance_boost": 0.05,
    },
    "arxiv_bio": {
        "label": "arXiv Biology",
        "url": "https://rss.arxiv.org/rss/q-bio",
        "type": "rss",
        "industry": "Biotech",
        "significance_boost": 0.05,
    },
    "hackernews": {
        "label": "Hacker News Top",
        "url": "https://hn.algolia.com/api/v1/search?tags=front_page&hitsPerPage=15",
        "type": "hn_api",
        "industry": "Tech",
        "significance_boost": 0.0,
    },
    "nature": {
        "label": "Nature Research",
        "url": "https://www.nature.com/nature.rss",
        "type": "rss",
        "industry": "Science",
        "significance_boost": 0.15,
    },
    "science_daily": {
        "label": "Science Daily",
        "url": "https://www.sciencedaily.com/rss/all.xml",
        "type": "rss",
        "industry": "Science",
        "significance_boost": 0.0,
    },
    "techcrunch": {
        "label": "TechCrunch",
        "url": "https://techcrunch.com/feed/",
        "type": "rss",
        "industry": "Startups",
        "significance_boost": 0.0,
    },
    "nasa": {
        "label": "NASA Breaking News",
        "url": "https://www.nasa.gov/rss/dyn/breaking_news.rss",
        "type": "rss",
        "industry": "Space",
        "significance_boost": 0.1,
    },
}

# Significance keywords — discoveries containing these get boosted
BREAKTHROUGH_KEYWORDS = [
    "breakthrough", "first-ever", "first ever", "novel", "revolutionary",
    "unprecedented", "discovery", "discovers", "discovered",
    "new method", "new approach", "state-of-the-art", "sota",
    "surpass", "outperform", "record-breaking", "world record",
    "cure", "vaccine", "quantum", "fusion", "superconductor",
    "general intelligence", "agi", "consciousness",
    "dark matter", "dark energy", "gravitational wave",
    "gene editing", "crispr", "mrna", "protein folding",
    "exoplanet", "habitable", "mars", "moon base",
    "neural interface", "brain-computer", "bci",
    "room temperature", "ambient", "zero-shot", "few-shot",
    "trillion", "billion parameter",
]


# ── Helpers ──────────────────────────────────────────────────

def _ensure_dir():
    DISCOVERY_DIR.mkdir(parents=True, exist_ok=True)


def _load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return DEFAULT_CONFIG.copy()


def _save_config(config: dict):
    _ensure_dir()
    CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")


def _load_seen() -> set:
    if SEEN_FILE.exists():
        try:
            return set(json.loads(SEEN_FILE.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass
    return set()


def _save_seen(seen: set):
    _ensure_dir()
    # Keep last 5000 hashes to prevent unbounded growth
    recent = list(seen)[-5000:]
    SEEN_FILE.write_text(json.dumps(recent), encoding="utf-8")


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _log_insight(entry: dict):
    _ensure_dir()
    with open(INSIGHTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _score_significance(title: str, summary: str, feed_boost: float) -> float:
    """Score how significant a discovery is (0.0 - 1.0)."""
    text = (title + " " + summary).lower()
    score = 0.2 + feed_boost  # Base score + feed source boost

    # Keyword matching
    hits = sum(1 for kw in BREAKTHROUGH_KEYWORDS if kw in text)
    score += min(0.4, hits * 0.08)

    # Title quality signals
    if any(w in title.lower() for w in ["breakthrough", "first", "novel", "new"]):
        score += 0.1
    if "?" not in title:  # Questions are usually less significant
        score += 0.02

    return min(1.0, round(score, 3))


# ── Feed Parsers ─────────────────────────────────────────────

def _parse_rss(content: str, max_items: int = 15) -> list[dict]:
    """Minimal RSS parser — no lxml dependency needed."""
    items = []
    # Extract <item> blocks
    item_blocks = re.findall(r'<item>(.*?)</item>', content, re.DOTALL)
    if not item_blocks:
        # Try <entry> (Atom format)
        item_blocks = re.findall(r'<entry>(.*?)</entry>', content, re.DOTALL)

    for block in item_blocks[:max_items]:
        title = _extract_tag(block, "title")
        link = _extract_tag(block, "link")
        if not link:
            # Atom format: <link href="..." />
            m = re.search(r'<link[^>]*href="([^"]*)"', block)
            if m:
                link = m.group(1)
        desc = _extract_tag(block, "description") or _extract_tag(block, "summary") or ""
        pub_date = _extract_tag(block, "pubDate") or _extract_tag(block, "published") or ""

        # Clean HTML from description
        desc = re.sub(r'<[^>]+>', '', desc).strip()
        desc = desc[:500]  # Truncate

        if title:
            items.append({
                "title": title.strip(),
                "link": link.strip() if link else "",
                "summary": desc,
                "published": pub_date.strip(),
            })

    return items


def _extract_tag(xml: str, tag: str) -> Optional[str]:
    """Extract content from an XML tag."""
    # Handle CDATA
    m = re.search(rf'<{tag}[^>]*>\s*<!\[CDATA\[(.*?)\]\]>\s*</{tag}>', xml, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(rf'<{tag}[^>]*>(.*?)</{tag}>', xml, re.DOTALL)
    if m:
        return m.group(1)
    return None


def _parse_hn(data: dict, max_items: int = 15) -> list[dict]:
    """Parse Hacker News Algolia API response."""
    items = []
    for hit in data.get("hits", [])[:max_items]:
        title = hit.get("title", "")
        url = hit.get("url", "") or f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}"
        points = hit.get("points", 0)
        comments = hit.get("num_comments", 0)

        if title:
            items.append({
                "title": title,
                "link": url,
                "summary": f"HN: {points} points, {comments} comments",
                "published": hit.get("created_at", ""),
            })

    return items


# ── Main Scraper ─────────────────────────────────────────────

def scrape_discoveries(brain=None, config: dict = None) -> dict:
    """
    Fetch discoveries from all enabled feeds.
    Store new ones in Watty's brain.
    Return summary of what was found.
    """
    if config is None:
        config = _load_config()

    seen = _load_seen()
    enabled = set(config.get("feeds_enabled", DEFAULT_CONFIG["feeds_enabled"]))
    max_per_feed = config.get("max_per_feed", 15)
    min_sig = config.get("min_significance", 0.3)
    custom_kw = config.get("custom_keywords", [])

    results = {
        "feeds_checked": 0,
        "discoveries_found": 0,
        "discoveries_stored": 0,
        "connections_found": 0,
        "errors": [],
        "top_discoveries": [],
    }

    all_discoveries = []

    for feed_id, feed in FEEDS.items():
        if feed_id not in enabled:
            continue

        results["feeds_checked"] += 1

        try:
            r = requests.get(feed["url"], timeout=15, headers={
                "User-Agent": "Watty/2.1 Brain Discovery Scraper"
            })
            r.raise_for_status()

            if feed["type"] == "rss":
                items = _parse_rss(r.text, max_per_feed)
            elif feed["type"] == "hn_api":
                items = _parse_hn(r.json(), max_per_feed)
            else:
                continue

            for item in items:
                h = _hash(item["title"])
                if h in seen:
                    continue

                seen.add(h)
                results["discoveries_found"] += 1

                # Score significance
                sig = _score_significance(
                    item["title"], item["summary"], feed["significance_boost"]
                )

                # Boost if matches custom keywords
                text_lower = (item["title"] + " " + item["summary"]).lower()
                for kw in custom_kw:
                    if kw.lower() in text_lower:
                        sig = min(1.0, sig + 0.15)

                if sig < min_sig:
                    continue

                discovery = {
                    "title": item["title"],
                    "link": item["link"],
                    "summary": item["summary"],
                    "source": feed["label"],
                    "industry": feed["industry"],
                    "significance": sig,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

                all_discoveries.append(discovery)

        except Exception as e:
            results["errors"].append(f"{feed['label']}: {str(e)[:100]}")

    # Sort by significance
    all_discoveries.sort(key=lambda d: d["significance"], reverse=True)

    # Store in brain
    if brain:
        for disc in all_discoveries:
            content = (
                f"[DISCOVERY] {disc['title']}\n"
                f"Source: {disc['source']} ({disc['industry']})\n"
                f"Significance: {disc['significance']}\n"
                f"{disc['summary']}\n"
                f"Link: {disc['link']}"
            )
            try:
                chunks = brain.store_memory(content, provider="discovery")
                results["discoveries_stored"] += 1

                # Check for connections to existing knowledge
                related = brain.recall(disc["title"], top_k=3)
                connections = [
                    r for r in related
                    if r["provider"] != "discovery" and r["score"] > 0.4
                ]
                if connections:
                    results["connections_found"] += len(connections)
                    disc["connections"] = [
                        {"content": c["content"][:100], "provider": c["provider"], "score": c["score"]}
                        for c in connections
                    ]
            except Exception as e:
                results["errors"].append(f"Store error: {str(e)[:80]}")

    # Log insights
    for disc in all_discoveries[:10]:
        _log_insight(disc)

    results["top_discoveries"] = [
        {"title": d["title"], "industry": d["industry"], "significance": d["significance"],
         "connections": len(d.get("connections", []))}
        for d in all_discoveries[:10]
    ]

    _save_seen(seen)
    return results


def get_recent_discoveries(n: int = 20) -> list[dict]:
    """Get recent discoveries from the log."""
    if not INSIGHTS_FILE.exists():
        return []
    lines = INSIGHTS_FILE.read_text(encoding="utf-8").strip().split("\n")
    entries = []
    for line in lines[-n:]:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries[::-1]  # Most recent first


def get_discovery_stats() -> dict:
    """Stats about discovery history."""
    if not INSIGHTS_FILE.exists():
        return {"total": 0, "industries": {}}
    lines = INSIGHTS_FILE.read_text(encoding="utf-8").strip().split("\n")
    industries = {}
    for line in lines:
        try:
            d = json.loads(line)
            ind = d.get("industry", "Unknown")
            industries[ind] = industries.get(ind, 0) + 1
        except json.JSONDecodeError:
            continue
    return {"total": len(lines), "industries": industries}


# ── MCP Tool Registration ────────────────────────────────────

from mcp.types import Tool, TextContent

TOOLS = [
    Tool(
        name="watty_discover",
        description=(
            "Watty's discovery radar. Scans the frontier of human knowledge "
            "across AI, biotech, physics, space, startups. One tool, four actions.\n"
            "Actions: scan (fetch latest), recent (view history), "
            "stats (coverage report), config (view/edit settings)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["scan", "recent", "stats", "config"],
                    "description": "Action to perform",
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "config: Set custom tracking keywords",
                },
                "n": {
                    "type": "integer",
                    "description": "recent: Number of discoveries to show (default: 10)",
                },
            },
            "required": ["action"],
        },
    ),
]


# ── Handlers ──────────────────────────────────────────────────

_brain_ref = None

def set_brain(brain):
    global _brain_ref
    _brain_ref = brain


async def handle_discover(arguments: dict) -> list[TextContent]:
    action = arguments.get("action", "stats")

    if action == "scan":
        result = scrape_discoveries(brain=_brain_ref)
        lines = [
            "Discovery scan complete.",
            f"  Feeds checked: {result['feeds_checked']}",
            f"  New discoveries: {result['discoveries_found']}",
            f"  Stored in brain: {result['discoveries_stored']}",
            f"  Connections to your work: {result['connections_found']}",
        ]
        if result["errors"]:
            lines.append(f"  Errors: {len(result['errors'])}")
        if result["top_discoveries"]:
            lines.append("\nTop discoveries:")
            for d in result["top_discoveries"][:5]:
                conn_tag = f" [{d['connections']} connections]" if d.get("connections") else ""
                lines.append(f"  [{d['significance']:.1f}] ({d['industry']}) {d['title'][:80]}{conn_tag}")
        return [TextContent(type="text", text="\n".join(lines))]

    elif action == "recent":
        n = arguments.get("n", 10)
        recent = get_recent_discoveries(n)
        if not recent:
            return [TextContent(type="text", text="No discoveries yet. Run scan first.")]
        lines = [f"Recent {len(recent)} discoveries:"]
        for d in recent:
            ts = d.get("timestamp", "?")[:10]
            lines.append(f"  [{ts}] ({d.get('industry', '?')}) {d.get('title', '?')[:80]}")
        return [TextContent(type="text", text="\n".join(lines))]

    elif action == "stats":
        stats = get_discovery_stats()
        lines = ["Discovery stats:", f"  Total tracked: {stats['total']}"]
        if stats["industries"]:
            lines.append("  By industry:")
            for ind, count in sorted(stats["industries"].items(), key=lambda x: x[1], reverse=True):
                lines.append(f"    {ind}: {count}")
        return [TextContent(type="text", text="\n".join(lines))]

    elif action == "config":
        config = _load_config()
        kw = arguments.get("keywords")
        if kw is not None:
            config["custom_keywords"] = kw
            _save_config(config)
            return [TextContent(type="text", text=f"Custom keywords set: {', '.join(kw)}")]
        return [TextContent(type="text", text=json.dumps(config, indent=2))]

    return [TextContent(type="text", text=f"Unknown action: {action}")]


HANDLERS = {
    "watty_discover": handle_discover,
}
