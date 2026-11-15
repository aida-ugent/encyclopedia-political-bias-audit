import pandas as pd
import mwapi
import mwparserfromhell
from datasets import load_dataset
import re
import time
import threading
import pickle
import unicodedata
from pathlib import Path
from typing import Optional, Tuple, List

from config import CONFIG, STOP_HEADINGS, logger
from article_collection.rate_limiter import rate_limiter
from article_collection.retry import retry_with_backoff
from article_collection.checkpoint_manager import CheckpointManager
from article_collection.parallel import process_parallel
from article_collection.grokipedia2_scraper.main import GrokipediaScraper

# Wikipedia API session
WIKI_API_SESSION = mwapi.Session(
    'https://en.wikipedia.org',
    user_agent='EncyclopediaAnalysis/1.0 (research project; contact@example.com) mwapi'
)

# Lazy loading for Grokipedia dataset (thread-safe)
_GROKI_DF = None
_GROKI_INDEX = None  # Dict for O(1) lookup
_GROKI_LOCK = threading.Lock()
_GROKI_INDEX_PATH = Path("data/cache/groki_index.pkl")

# Grokipedia 2.0 scraper instance
_GROKI2_SCRAPER = GrokipediaScraper()


def save_groki_index(index: dict, path: Path = None) -> None:
    """Save the Grokipedia index to disk as a pickle file."""
    path = path or _GROKI_INDEX_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"Saved Grokipedia index ({len(index)} titles) to {path}")


def load_groki_index(path: Path = None) -> Optional[dict]:
    """Load the Grokipedia index from a pickle file on disk.

    Returns the index dict, or None if the file doesn't exist or is corrupt.
    """
    path = path or _GROKI_INDEX_PATH
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            index = pickle.load(f)
        logger.info(f"Loaded Grokipedia index ({len(index)} titles) from {path}")
        return index
    except Exception as e:
        logger.warning(f"Failed to load Grokipedia index from {path}: {e}")
        return None


def get_groki_df():
    """Lazy load Grokipedia dataset (thread-safe).

    Tries to load a cached index from disk first. If unavailable, downloads
    the dataset from HuggingFace, builds the index, and saves it for next time.
    """
    global _GROKI_DF, _GROKI_INDEX
    if _GROKI_DF is None:
        with _GROKI_LOCK:
            if _GROKI_DF is None:
                # Try loading cached index first
                cached_index = load_groki_index()
                if cached_index is not None:
                    _GROKI_INDEX = cached_index
                    # Load a minimal DataFrame (index is sufficient for lookups)
                    _GROKI_DF = pd.DataFrame(columns=["title", "text", "title_normalized"])
                    logger.info("Using cached index — skipped full dataset download")
                else:
                    logger.info("Loading Grokipedia dataset...")
                    _GROKI_DF = load_dataset("htriedman/grokipedia-v0.1-dump")["train"].to_pandas()
                    _GROKI_DF["title_normalized"] = _GROKI_DF["title"].apply(normalize)

                    # Build index: normalized_title -> list of texts
                    logger.info("Building title index...")
                    _GROKI_INDEX = {}
                    for _, row in _GROKI_DF.iterrows():
                        key = row["title_normalized"]
                        if key not in _GROKI_INDEX:
                            _GROKI_INDEX[key] = []
                        _GROKI_INDEX[key].append(row["text"])

                    logger.info(f"Loaded {len(_GROKI_DF)} Grokipedia articles, indexed {len(_GROKI_INDEX)} unique titles")

                    # Save index to disk for future runs
                    save_groki_index(_GROKI_INDEX)
    return _GROKI_DF, _GROKI_INDEX


def clean_text(text):
    text = str(text)
    text = re.sub(r"\[\d+\]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize(text):
    return re.sub(r'[^A-Za-z0-9]', '', str(text)).lower()


def get_groki_text(title: str) -> str:
    """Get text from Grokipedia 1.0 dataset using O(1) index lookup."""
    _, groki_index = get_groki_df()
    cleaned_title = normalize(title)

    # O(1) lookup instead of scanning 45M rows
    texts = groki_index.get(cleaned_title, [])
    return "\n".join(texts)


def get_groki2_text(title: str) -> str:
    """Get text from Grokipedia 2.0 via web scraper."""
    try:
        text = _GROKI2_SCRAPER.scrape_article(title)
        return text if text else ""
    except Exception as e:
        logger.error(f"Failed to scrape Grokipedia 2.0 for '{title}': {e}")
        return ""

_POLITICIAN_KEYWORDS = [
    "politician", "political", "senator", "governor", "president",
    "prime minister", "minister", "congressman", "congresswoman",
    "representative", "member of parliament", "mayor", "chancellor",
    "secretary of state", "legislator", "lawmaker", "statesman",
    "stateswoman", "cabinet", "parliament", "congress", "senate",
    "house of representatives", "elected", "party leader", "chair",
    "public official", "government official", "chairperson", "chairman",
    "secretary", "commander", "chairwoman"
]

_DISAMBIG_MARKERS = ("{{disambiguation", "{{disambig", "{{hndis", "{{human name disambiguation")
_LIST_MARKERS = ("{{listof", "{{list of", "[[category:lists", "[[category:list")


# ── Wikipedia API helpers ──────────────────────────────────────────────────────

@retry_with_backoff()
def _fetch_wikipedia_content(title: str) -> dict:
    """Fetch Wikipedia content with rate limiting and retry."""
    rate_limiter.wait_if_needed()
    return WIKI_API_SESSION.get(
        action="query",
        prop="revisions",
        rvprop="content",
        rvslots="main",
        titles=title,
        redirects=True,
        formatversion=2,
    )


@retry_with_backoff()
def _search_wikipedia(query: str, limit: int = 1) -> dict:
    """Search Wikipedia with rate limiting and retry."""
    rate_limiter.wait_if_needed()
    return WIKI_API_SESSION.get(
        action="query",
        list="search",
        srsearch=query,
        srlimit=limit,
    )


def _try_fetch_article(title: str) -> Tuple[Optional[str], Optional[str]]:
    """Fetch a Wikipedia article by exact title. Returns (resolved_title, wikitext)."""
    try:
        resp = _fetch_wikipedia_content(title)
        pages = resp.get("query", {}).get("pages", [])
        if not pages or "revisions" not in pages[0]:
            return None, None
        resolved = pages[0].get("title", title)
        wikitext = pages[0]["revisions"][0]["slots"]["main"]["content"]
        return resolved, wikitext
    except Exception:
        return None, None


# ── Name normalisation ────────────────────────────────────────────────────────

def _is_middle_initial(part: str, index: int, num_parts: int) -> bool:
    """True if *part* is a single-letter middle initial (not first/last name)."""
    return 0 < index < num_parts - 1 and len(part.rstrip(".")) == 1 and part[0].isalpha()


def _name_variants(name: str) -> List[str]:
    """Name forms to compare against search results (no API calls).

    Returns normalised strings like ["john smith", "john a smith", "john a. smith"].
    """
    name = name.strip()
    parts = name.split()
    variants = [name.lower()]

    if len(parts) >= 3:
        toggled = [
            (p + "." if len(p) == 1 else p[0]) if _is_middle_initial(p, i, len(parts)) else p
            for i, p in enumerate(parts)
        ]
        toggled_name = " ".join(toggled).lower()
        if toggled_name not in variants:
            variants.append(toggled_name)

        short = " ".join(p for i, p in enumerate(parts) if not _is_middle_initial(p, i, len(parts))).lower()
        if short not in variants:
            variants.append(short)

    return variants


# ── Page classification ───────────────────────────────────────────────────────

def _is_disambiguation_page(wikitext: str) -> bool:
    lower = wikitext.lower()
    return any(m in lower for m in _DISAMBIG_MARKERS)


def _is_list_page(title: str, wikitext: str) -> bool:
    if title.lower().startswith(("list of ", "lists of ")):
        return True
    return any(m in wikitext[:2000].lower() for m in _LIST_MARKERS)


def _is_valid_article(title: str, wikitext: str) -> bool:
    """Neither a disambiguation page nor a list page."""
    return not _is_disambiguation_page(wikitext) and not _is_list_page(title, wikitext)


def _is_politician_page(wikitext: str) -> bool:
    lower = wikitext[:3000].lower()
    return any(re.search(rf"\b{re.escape(kw)}\b", lower) for kw in _POLITICIAN_KEYWORDS)


# ── Candidate ranking ────────────────────────────────────────────────────────
def _strip_diacritics(text: str) -> str:
    """Remove diacritics/accents from text (e.g. 'Fernández' -> 'Fernandez')."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _score_candidate(candidate_title: str, name_variants: List[str], name_parts: List[str]) -> float:
    """Score a search-result title for how well it matches a politician name.

    Higher is better.  Components:
      +3.0 — exact name-variant match (after stripping disambiguator)
      +1.0 — "politician" appears in the title
      +0.0–1.0 — fraction of name parts found in the title
    """
    title_lower = _strip_diacritics(candidate_title.lower())
    base = re.sub(r"\s*\(.*\)\s*$", "", title_lower).strip()

    exact = re.split(r"[\s-]+", base) == re.split(r"[\s-]+", name_variants[0]) if name_variants else False
    if not exact:
        base_words = re.split(r"[\s-]+", base)
        exact = any(base_words == re.split(r"[\s-]+", v) for v in name_variants)
    has_politician = "politician" in title_lower
    parts_found = sum(1 for p in name_parts if re.search(rf"\b{re.escape(p)}\b", base))
    part_ratio = parts_found / len(name_parts) if name_parts else 0.0

    score = part_ratio
    if exact:
        score += 3.0
    if has_politician:
        score += 0.3
    return score


# ── Disambiguation resolution ─────────────────────────────────────────────────
def _resolve_disambiguation(wikitext: str, name_variants: List[str], name_parts: List[str]) -> List[str]:
    """Extract candidate titles from a disambiguation page, ranked by relevance."""
    wikicode = mwparserfromhell.parse(wikitext)
    candidates = []

    for link in wikicode.filter_wikilinks():
        link_str = str(link)
        pos = wikitext.find(link_str)
        if pos == -1:
            continue
        line_start = wikitext.rfind("\n", 0, pos)
        line_end = wikitext.find("\n", pos)
        context = wikitext[line_start : line_end if line_end != -1 else len(wikitext)].lower()

        if _is_politician_page(context):
            title = str(link.title).strip()
            if title not in candidates:
                candidates.append(title)

    candidates.sort(key=lambda c: _score_candidate(c, name_variants, name_parts), reverse=True)
    return candidates


# ── Article text extraction ───────────────────────────────────────────────────

def _extract_article_text(wikitext: str) -> str:
    """Parse wikitext, strip content after stop headings, return clean text."""
    # Pre-strip <ref> tags before parsing — mwparserfromhell sometimes fails
    # to recognise them (malformed / nested), leaving citation text in output.
    wikitext = re.sub(r"<ref[^>]*>.*?</ref>", "", wikitext, flags=re.DOTALL)
    wikitext = re.sub(r"<ref[^>]*/>", "", wikitext)
    wikicode = mwparserfromhell.parse(wikitext)
    for node in wikicode.nodes:
        if node.__class__.__name__ == "Heading" and node.title.strip() in STOP_HEADINGS:
            wikicode.nodes = wikicode.nodes[: wikicode.nodes.index(node)]
            break
    text = wikicode.strip_code(normalize=True, collapse=True)
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


# ── Fetch & validate a single candidate ───────────────────────────────────────

def _try_resolve_candidate(
    title: str, name_variants: List[str], name_parts: List[str]
) -> Tuple[Optional[str], Optional[str]]:
    """Fetch *title*, handle disambiguation, validate as politician article.

    Returns (article_title, extracted_text) or (None, None).
    """
    article_title, wikitext = _try_fetch_article(title)
    if not wikitext:
        return None, None

    if _is_disambiguation_page(wikitext):
        logger.info(f"'{title}' is a disambiguation page, resolving...")
        ranked = _resolve_disambiguation(wikitext, name_variants, name_parts)
        for candidate in ranked:
            logger.info(f"Trying disambiguation candidate '{candidate}'")
            article_title, wikitext = _try_fetch_article(candidate)
            if not wikitext or not _is_valid_article(article_title, wikitext):
                continue
            wikitext = _extract_article_text(wikitext)
            if _is_politician_page(wikitext):
                return article_title, wikitext
            logger.info(f"'{candidate}' is not a politician page")
        return None, None

    if not _is_valid_article(article_title, wikitext):
        return None, None

    wikitext = _extract_article_text(wikitext)
    if _is_politician_page(wikitext):
        return article_title, wikitext

    logger.info(f"'{title}' -> '{article_title}' doesn't appear to be a politician page")
    return None, None


# ── Main entry point ──────────────────────────────────────────────────────────

def get_wikipedia_article(title: str) -> Tuple[Optional[str], Optional[str]]:
    """Fetch Wikipedia article content for a politician by name.

    Uses a single search API call to retrieve candidate titles, scores them
    by how well they match the input name + politician heuristics, then
    fetches only the best candidate(s) until one succeeds.
    """
    try:
        name_variants = _name_variants(title)
        name_parts = [p.lower() for p in re.split(r"[\s-]+", title) if p]

        search_resp = _search_wikipedia(f"{title}", limit=10)
        hits = search_resp.get("query", {}).get("search", [])

        # Score and sort candidates (best match first)
        scored = sorted(
            hits,
            key=lambda h: _score_candidate(h["title"], name_variants, name_parts),
            reverse=True,
        )

        for hit in scored:
            score = _score_candidate(hit["title"], name_variants, name_parts)
            if score == 0:
                continue
            result = _try_resolve_candidate(hit["title"], name_variants, name_parts)
            if result[0]:
                if score >= 3:
                    logger.info(f"'{title}' -> '{result[0]}' (score {score})")
                else:
                    logger.info(f"'{title}' -> '{result[0]}' (score {score}, weaker match)")
                return result

        logger.warning(f"Could not find politician article for '{title}'")
        return None, None

    except Exception as e:
        logger.error(f"Failed to fetch '{title}': {e}")
        return None, None


def process_article(title: str) -> Optional[dict]:
    """Process a single article."""
    wiki_title, wiki_text = get_wikipedia_article(title)
    if not wiki_text:
        logger.warning(f'Wikipedia page for "{title}" not found')
        return None

    # Grokipedia 0.1 (from HuggingFace dataset)
    groki_text = get_groki_text(title)
    if not groki_text and wiki_title != title:
        groki_text = get_groki_text(wiki_title)

    if not groki_text:
        logger.warning(f'Grokipedia 0.1 page for "{title}" not found')

    # Grokipedia 0.2 (from web scraper)
    groki2_url_attempts = [title]
    if wiki_title and wiki_title != title:
        groki2_url_attempts.append(wiki_title)
    if wiki_title and wiki_title != wiki_title.lower():
        groki2_url_attempts.append(wiki_title.lower())
    if "-" in wiki_title:
        groki2_url_attempts.append(wiki_title.replace("-", "_"))
    if not wiki_title.isascii():
        ascii_title = wiki_title.encode('ascii', 'ignore').decode('ascii')
        groki2_url_attempts.append(ascii_title)
        if "-" in ascii_title:
            groki2_url_attempts.append(ascii_title.replace("-", "_"))
        if ascii_title.lower() != ascii_title:
            groki2_url_attempts.append(ascii_title.lower())


    for title_attempt in groki2_url_attempts:
        groki2_text = get_groki2_text(title_attempt)
        if groki2_text:
            break

    if not groki_text and not groki2_text:
        logger.warning(f'Grokipedia 0.1 and 0.2 pages for "{title}" not found')
        return None

    return {
        "name": title,
        "wikipedia": clean_text(wiki_text),
        "grokipedia_0_1": clean_text(groki_text),
        "grokipedia_0_2": clean_text(groki2_text)
    }


def collect_articles(
    input_file: str = "LeadersAndParties/leaders.csv", output_file: str = None, resume: bool = True
):
    """
    Collect Wikipedia and Grokipedia articles.

    Args:
        input_file: Path relative to data/ directory (.txt or .csv)
        output_file: Output path (default: data/articles/<input_file>)
        resume: Whether to resume from checkpoint
    """
    input_path = Path(f'data/{input_file}')
    df = pd.read_csv(input_path)
    # df = df.head(50)  # For testing, remove or adjust as needed
    article_titles = df["name"].dropna().unique().tolist()
    logger.info(f"Loaded {len(article_titles)} article titles")

    # Set up output and checkpoint
    output_path = Path(output_file or f'data/articles/{input_file}')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = CheckpointManager(str(output_path.with_suffix('.checkpoint.json')))

    # Filter already processed if resuming
    if resume and checkpoint.processed_titles:
        original = len(article_titles)
        article_titles = [t for t in article_titles if not checkpoint.is_processed(t)]
        logger.info(f"Resuming: {original - len(article_titles)} done, {len(article_titles)} remaining")

    # Create output file with header if needed
    if not resume or not output_path.exists():
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("name,wikipedia,grokipedia_0_1,grokipedia_0_2\n")

    # Process in batches
    batch_size = CONFIG["batch_size"]
    batches = [article_titles[i:i + batch_size] for i in range(0, len(article_titles), batch_size)]

    collected, failed = 0, 0
    start_time = time.time()

    try:
        for batch_num, batch in enumerate(batches, 1):
            logger.info(f"Starting batch {batch_num}/{len(batches)} ({len(batch)} articles)")

            # Process batch in parallel
            results = process_parallel(batch, process_article, batch_num=batch_num, total_batches=len(batches))

            # Save results
            for title, result in results:
                pd.DataFrame([result]).to_csv(output_path, mode='a', header=False, index=False)
                collected += 1

            # Mark all as processed
            successful_titles = {title for title, _ in results}
            for title in batch:
                checkpoint.mark_processed(title)
                if title not in successful_titles:
                    failed += 1

            checkpoint.save()

            if batch_num < len(batches):
                time.sleep(CONFIG["batch_delay"])

    except KeyboardInterrupt:
        logger.warning("Interrupted. Checkpoint saved.")
        checkpoint.save()
        raise

    # Summary
    elapsed = time.time() - start_time
    logger.info(f"Done! Collected: {collected}, Failed: {failed}, Time: {elapsed:.0f}s")

    if failed == 0:
        checkpoint.clear()


def main():
    collect_articles(
        input_file="LeadersAndParties/leaders.csv",
        output_file=None,  # Uses default: data/articles/LeadersAndParties/leaders.csv
        resume=True
    )


if __name__ == "__main__":
    main()
