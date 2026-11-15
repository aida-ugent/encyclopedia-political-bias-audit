"""
Grokipedia Scraper

A tool to extract and format text content from Grokipedia articles.
"""

import re
import argparse
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup


class GrokipediaScraper:
    """Scraper for Grokipedia articles."""

    BASE_URL = "https://grokipedia.com/page"

    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the scraper.

        Args:
            output_dir: Directory to save output files. Defaults to current directory.
        """
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "test"
        })

    @staticmethod
    def _clean_text(text: str) -> str:
        """Normalise whitespace produced by get_text(separator=' ')."""
        # Collapse runs of whitespace (but not newlines) into a single space
        text = re.sub(r"[^\S\n]+", " ", text)
        # Remove spaces before punctuation introduced by HTML inline splits
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        return text.strip()

    def _build_url(self, article_name: str) -> str:
        """
        Build the full URL for an article.

        Args:
            article_name: Name of the article (e.g., 'Elon_Musk' or 'Elon Musk')

        Returns:
            Full URL to the article
        """
        # Replace spaces with underscores for URL compatibility
        formatted_name = article_name.replace(" ", "_")
        return f"{self.BASE_URL}/{formatted_name}"

    def _fetch_page(self, url: str) -> Optional[str]:
        """
        Fetch the HTML content of a page.

        Args:
            url: URL to fetch

        Returns:
            HTML content as string, or None if failed
        """
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            # Grokipedia pages are UTF-8; decode from raw bytes explicitly
            # to avoid mojibake like "â" instead of "—".
            try:
                return response.content.decode("utf-8")
            except UnicodeDecodeError:
                # Fallback to Requests' detected encoding if strict UTF-8 fails.
                return response.text
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    def _extract_text(self, html: str) -> Optional[str]:
        """
        Extract and format text content from HTML.

        Args:
            html: Raw HTML content

        Returns:
            Formatted plain text content, or None if article not found
        """
        soup = BeautifulSoup(html, "html.parser")

        # Check for 404 "Article not found" page
        h1 = soup.find("h1")
        if h1 and "Article not found" in h1.get_text():
            return None

        # Also check for the specific "doesn't exist yet" message
        if "This article doesn't exist yet in Grokipedia" in html:
            return None

        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header", "button", "svg"]):
            element.decompose()

        # Find the main content area - Grokipedia uses article content structure
        # Try to find the main article content
        main_content = soup.find("main") or soup.find("article") or soup

        # Extract title (used only for duplicate-removal logic, not output)
        title_elem = main_content.find("h1")
        title = title_elem.get_text(strip=True) if title_elem else ""

        # Build the text content
        text_parts = []

        # Track seen texts to avoid duplicates
        seen_texts = set()

        # Grokipedia uses spans with 'block' class for paragraph text
        # and standard h1-h6 for headers, ul/ol for lists
        # Process elements in document order
        content_elements = main_content.find_all(
            ["h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "blockquote", "span"]
        )

        for element in content_elements:
            # Skip all h1 elements (article title should not be included)
            if element.name == "h1":
                continue

            # For spans, only process those that look like paragraph blocks
            # (have 'block' in their class list and contain substantial text)
            if element.name == "span":
                classes = element.get("class") or []
                if "block" not in classes:
                    continue
                # Skip spans that are inside other spans we'll process
                # (i.e., only process top-level block spans)
                parent_span = element.find_parent("span", class_=lambda x: x and "block" in x)
                if parent_span:
                    continue

            text = self._clean_text(element.get_text(separator=" "))

            # Skip empty or very short text
            if not text or len(text) < 15:
                continue

            # Skip reference URLs and citations (common in references section)
            if text.startswith("http") or text.startswith("https"):
                continue

            # Skip if we've already seen this exact text
            if text in seen_texts:
                continue
            seen_texts.add(text)

            # Strip leading title from text when the element contains the
            # <h1> as a descendant (avoids duplication from a parent
            # element's get_text including the title)
            if title and text.startswith(title) and title_elem in element.descendants:
                text = text[len(title):].lstrip()
                if not text or len(text) < 15:
                    continue

            # Format based on element type
            if element.name == "h2":
                text_parts.append(f"\n## {text}\n")
            elif element.name == "h3":
                text_parts.append(f"\n### {text}\n")
            elif element.name == "h4":
                text_parts.append(f"\n#### {text}\n")
            elif element.name in ("h5", "h6"):
                text_parts.append(f"\n##### {text}\n")
            elif element.name == "blockquote":
                text_parts.append(f"\n> {text}\n")
            elif element.name in ("ul", "ol"):
                # Process list items
                items = []
                for li in element.find_all("li", recursive=False):
                    li_text = self._clean_text(li.get_text(separator=" "))
                    if li_text and len(li_text) > 5:
                        items.append(f"• {li_text}")
                        seen_texts.add(li_text)
                if items:
                    text_parts.append("\n".join(items) + "\n")
            else:
                # Paragraphs (p tags and block spans)
                text_parts.append(text + "\n")

        # Join and clean up the text
        text = "\n".join(text_parts)

        # Clean up excessive newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove citation numbers like [1], [2], etc.
        text = re.sub(r"\[\d+\]", "", text)

        # Clean up any remaining whitespace issues
        text = text.strip()

        return text

    def scrape_article(self, article_name: str) -> Optional[str]:
        """
        Scrape a single article and return its text content.

        Args:
            article_name: Name of the article to scrape

        Returns:
            Formatted text content, or None if failed
        """
        url = self._build_url(article_name)
        print(f"Fetching: {url}")

        html = self._fetch_page(url)
        if not html:
            return None

        text = self._extract_text(html)
        return text

    def scrape_and_save(self, article_name: str) -> Optional[Path]:
        """
        Scrape an article and save it to a text file.

        Args:
            article_name: Name of the article to scrape

        Returns:
            Path to the saved file, or None if failed
        """
        text = self.scrape_article(article_name)
        if not text:
            print(f"Failed to scrape article: {article_name}")
            return None

        # Create filename from article name
        safe_filename = article_name.replace(" ", "_").replace("/", "_")
        output_path = self.output_dir / f"{safe_filename}.txt"

        # Save to file
        output_path.write_text(text, encoding="utf-8")
        print(f"Saved to: {output_path}")

        return output_path


def main():
    """Command-line interface for the Grokipedia scraper."""
    parser = argparse.ArgumentParser(
        description="Scrape articles from Grokipedia and save as text files."
    )
    parser.add_argument(
        "article",
        type=str,
        help="Name of the article to scrape (e.g., 'Elon_Musk' or 'Elon Musk')"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default=None,
        help="Directory to save output files (default: current directory)"
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the text to stdout instead of saving to file"
    )

    args = parser.parse_args()

    scraper = GrokipediaScraper(output_dir=args.output_dir)

    if args.print_only:
        text = scraper.scrape_article(args.article)
        if text:
            print(text)
    else:
        scraper.scrape_and_save(args.article)


if __name__ == "__main__":
    main()
