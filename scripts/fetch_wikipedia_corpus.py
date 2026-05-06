"""Fetch Wikipedia articles as lecture corpus for LectureRAG-75 domains.

Alternative to YouTube transcripts — uses Wikipedia's REST API (no auth, no
rate limiting, ~10-50 KB per article). Saves merged text to
data/lecture_rag_75/transcripts/<domain>/<domain>.txt

Usage:
    python scripts/fetch_wikipedia_corpus.py
    python scripts/fetch_wikipedia_corpus.py --domain machine_learning
"""

import argparse
import json
import os
import re
import time
import requests

TRANSCRIPTS_DIR = "data/lecture_rag_75/transcripts"
DELAY = 0.5  # seconds between requests — Wikipedia asks for politeness

DOMAIN_ARTICLES = {
    "machine_learning": [
        "Machine_learning",
        "Supervised_learning",
        "Unsupervised_learning",
        "Neural_network_(machine_learning)",
        "Gradient_descent",
        "Overfitting",
        "Cross-validation_(statistics)",
        "Decision_tree_learning",
        "Support_vector_machine",
        "Naive_Bayes_classifier",
    ],
    "computer_networks": [
        "Computer_network",
        "OSI_model",
        "Internet_protocol_suite",
        "Transmission_Control_Protocol",
        "IP_address",
        "Domain_Name_System",
        "Network_switch",
        "Router_(computing)",
        "Ethernet",
        "Firewall_(computing)",
    ],
    "database_systems": [
        "Database",
        "Relational_database",
        "SQL",
        "Database_normalization",
        "Database_index",
        "ACID_(atomicity,_consistency,_isolation,_durability)",
        "Entity–relationship_model",
        "NoSQL",
        "Transaction_processing",
        "Query_optimization",
    ],
    "operating_systems": [
        "Operating_system",
        "Process_(computing)",
        "Thread_(computing)",
        "Scheduling_(computing)",
        "Virtual_memory",
        "File_system",
        "Deadlock",
        "Semaphore_(programming)",
        "Memory_management",
        "System_call",
    ],
}

HEADERS = {
    "User-Agent": "LectureRAG-Research/1.0 (educational research project; contact: research@example.com)",
    "Accept": "application/json",
}


def fetch_article_text(title: str, max_chars: int = 8000) -> str:
    """Fetch plain-text content of a Wikipedia article via the MediaWiki API."""
    api_url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "titles": title.replace("_", " "),
        "format": "json",
        "explaintext": True,
        "exsectionformat": "plain",
        "exlimit": 1,
    }
    try:
        r = requests.get(api_url, params=params, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            pages = r.json().get("query", {}).get("pages", {})
            for page in pages.values():
                extract = page.get("extract", "").strip()
                if extract:
                    extract = re.sub(r"\n{3,}", "\n\n", extract)
                    return extract[:max_chars]
    except Exception as e:
        print(f"    [mediawiki error] {e}")
    return ""


def build_domain_corpus(domain: str, articles: list[str], out_dir: str) -> int:
    domain_dir = os.path.join(out_dir, domain)
    os.makedirs(domain_dir, exist_ok=True)
    out_path = os.path.join(domain_dir, f"{domain}.txt")

    if os.path.exists(out_path):
        size = os.path.getsize(out_path)
        print(f"  [skip] {out_path} already exists ({size} bytes)")
        return size

    texts = []
    for title in articles:
        print(f"  Fetching: {title} ...", end=" ", flush=True)
        text = fetch_article_text(title)
        if text:
            texts.append(f"# {title.replace('_', ' ')}\n\n{text}")
            print(f"OK ({len(text)} chars)")
        else:
            print("EMPTY")
        time.sleep(DELAY)

    if not texts:
        print(f"  [warn] No content fetched for {domain}")
        return 0

    merged = "\n\n".join(texts)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(merged)

    print(f"  Saved -> {out_path}  ({len(merged)} chars, {len(texts)}/{len(articles)} articles)")
    return len(merged)


def main():
    parser = argparse.ArgumentParser(description="Fetch Wikipedia corpus for LectureRAG-75")
    parser.add_argument("--domain", default=None, choices=list(DOMAIN_ARTICLES.keys()),
                        help="Single domain to fetch (default: all 4 new domains)")
    parser.add_argument("--out", default=TRANSCRIPTS_DIR)
    args = parser.parse_args()

    domains = {args.domain: DOMAIN_ARTICLES[args.domain]} if args.domain else DOMAIN_ARTICLES

    total_chars = 0
    for domain, articles in domains.items():
        print(f"\nDomain: {domain}  ({len(articles)} articles)")
        total_chars += build_domain_corpus(domain, articles, args.out)

    print(f"\nDone. Total corpus: {total_chars:,} chars across {len(domains)} domain(s).")
    print("Next: python scripts/build_domain_stores.py")


if __name__ == "__main__":
    main()
