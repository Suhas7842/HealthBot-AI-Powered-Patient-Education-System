"""
PubMed medical article loader using Biopython Entrez API.
Fetches articles for common medical conditions to build knowledge base.
"""

import time
from typing import List, Dict, Optional
from Bio import Entrez
import pandas as pd
from tqdm import tqdm
from healthbot.logger import logger

# NCBI requires email for API access
Entrez.email = "healthbot@example.com"

# Common medical conditions to fetch
MEDICAL_CONDITIONS = [
    "diabetes mellitus",
    "hypertension",
    "asthma",
    "heart disease",
    "coronary artery disease",
    "depression",
    "anxiety disorder",
    "arthritis",
    "migraine",
    "allergies",
    "influenza",
    "common cold",
    "obesity",
    "cancer",
    "stroke",
    "chronic obstructive pulmonary disease",
    "pneumonia",
    "gastritis",
    "chronic kidney disease",
    "thyroid disorder",
]


class PubMedLoader:
    """Fetches medical articles from PubMed for building knowledge base."""

    def __init__(self, articles_per_condition: int = 50, rate_limit_delay: float = 0.34):
        """
        Initialize PubMed loader.

        Args:
            articles_per_condition: Number of articles to fetch per condition
            rate_limit_delay: Delay between requests (3 req/sec = 0.34s delay)
        """
        self.articles_per_condition = articles_per_condition
        self.rate_limit_delay = rate_limit_delay
        self.all_articles: List[Dict] = []

    def search_pubmed(self, query: str, max_results: int) -> List[str]:
        """
        Search PubMed and return list of PMIDs.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of PubMed IDs (PMIDs)
        """
        try:
            handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results,
                sort="relevance"
            )
            record = Entrez.read(handle)
            handle.close()
            return record["IdList"]
        except Exception as e:
            logger.error(f"PubMed search failed for '{query}': {e}")
            return []

    def fetch_article_details(self, pmid: str) -> Optional[Dict]:
        """
        Fetch full article details for a given PMID.

        Args:
            pmid: PubMed ID

        Returns:
            Dictionary with article details or None if fetch fails
        """
        try:
            handle = Entrez.efetch(
                db="pubmed",
                id=pmid,
                rettype="medline",
                retmode="text"
            )
            record = handle.read()
            handle.close()

            # Parse the medline format
            article_data = self._parse_medline(record, pmid)
            return article_data

        except Exception as e:
            logger.error(f"Failed to fetch PMID {pmid}: {e}")
            return None

    def _parse_medline(self, medline_text: str, pmid: str) -> Dict:
        """
        Parse medline format text into structured data.

        Args:
            medline_text: Raw medline format text
            pmid: PubMed ID

        Returns:
            Dictionary with parsed article data
        """
        lines = medline_text.split("\n")
        title = ""
        abstract = ""
        authors = []
        journal = ""
        year = ""

        current_field = None
        for line in lines:
            if line.startswith("TI  - "):
                current_field = "title"
                title = line[6:].strip()
            elif line.startswith("AB  - "):
                current_field = "abstract"
                abstract = line[6:].strip()
            elif line.startswith("AU  - "):
                authors.append(line[6:].strip())
            elif line.startswith("TA  - "):
                journal = line[6:].strip()
            elif line.startswith("DP  - "):
                year = line[6:].split()[0] if line[6:] else ""
            elif line.startswith("      ") and current_field:
                # Continuation of previous field
                if current_field == "title":
                    title += " " + line.strip()
                elif current_field == "abstract":
                    abstract += " " + line.strip()

        return {
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "authors": "; ".join(authors[:3]),  # First 3 authors
            "journal": journal,
            "year": year,
        }

    def fetch_articles_for_condition(self, condition: str) -> List[Dict]:
        """
        Fetch articles for a specific medical condition.

        Args:
            condition: Medical condition to search

        Returns:
            List of article dictionaries
        """
        logger.info(f"Fetching articles for: {condition}")

        # Search for PMIDs
        pmids = self.search_pubmed(condition, self.articles_per_condition)
        time.sleep(self.rate_limit_delay)

        if not pmids:
            logger.warning(f"No articles found for: {condition}")
            return []

        articles = []
        for pmid in tqdm(pmids, desc=f"Fetching {condition}", leave=False):
            article = self.fetch_article_details(pmid)
            if article and article["abstract"]:  # Only keep articles with abstracts
                article["condition"] = condition
                articles.append(article)
            time.sleep(self.rate_limit_delay)  # Rate limiting

        logger.info(f"Fetched {len(articles)} articles for {condition}")
        return articles

    def build_knowledge_base(
        self,
        conditions: Optional[List[str]] = None,
        output_path: str = "data/medical_kb.parquet"
    ) -> pd.DataFrame:
        """
        Build complete medical knowledge base by fetching articles for all conditions.

        Args:
            conditions: List of conditions to fetch (defaults to MEDICAL_CONDITIONS)
            output_path: Path to save the knowledge base

        Returns:
            DataFrame with all fetched articles
        """
        if conditions is None:
            conditions = MEDICAL_CONDITIONS

        logger.info(f"Building knowledge base for {len(conditions)} conditions")
        logger.info(f"Target: {self.articles_per_condition} articles per condition")

        all_articles = []
        for condition in tqdm(conditions, desc="Processing conditions"):
            articles = self.fetch_articles_for_condition(condition)
            all_articles.extend(articles)

        # Convert to DataFrame
        df = pd.DataFrame(all_articles)

        # Save to parquet
        df.to_parquet(output_path, index=False)
        logger.info(f"Knowledge base saved to {output_path}")
        logger.info(f"Total articles: {len(df)}")
        logger.info(f"Conditions covered: {df['condition'].nunique()}")

        return df


def main():
    """Main function to build knowledge base."""
    import os

    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Initialize loader
    loader = PubMedLoader(articles_per_condition=50, rate_limit_delay=0.34)

    # Build knowledge base
    df = loader.build_knowledge_base()

    # Print summary statistics
    print("\n" + "="*80)
    print("KNOWLEDGE BASE SUMMARY")
    print("="*80)
    print(f"Total articles: {len(df)}")
    print(f"Conditions: {df['condition'].nunique()}")
    print("\nArticles per condition:")
    print(df['condition'].value_counts().to_string())
    print("="*80)


if __name__ == "__main__":
    main()
