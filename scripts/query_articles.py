import argparse
import pandas as pd
import requests
from dotenv import load_dotenv
import os
import json

load_dotenv()


def query_api(params):
    url = "https://newsapi.org/v2/everything"
    response = requests.get(url, params=params)
    response.raise_for_status()
    articles = json.loads(response.text)["articles"] or []
    df = pd.DataFrame.from_dict(articles, orient="columns")
    df.reset_index(level=0, inplace=True)

    if not df.empty:
        df["source"] = df["source"].apply(lambda x: x["name"])

    return df

def query_articles(
    query,
    api_key,
    output=None,
    sort_by="publishedAt",
    search_in="title",
    from_date=None,
    to_date=None,
    sources=None,
    exclude_domains=None,
    nb_pages=1,
):
    """Query articles from News API. Write to CSV if output is provided. Returns DataFrame."""
    results = pd.DataFrame()

    for page in range(1, nb_pages + 1):
        result = query_api(
            {
                "q": query,
                "apiKey": api_key,
                "from": from_date,
                "to": to_date,
                "language": "en",
                "sortBy": sort_by,
                "searchIn": search_in,
                "sources": sources,
                "excludeDomains": exclude_domains,
                "page": page,
            }
        )

        results = pd.concat([results, result], ignore_index=True)

    if output:
        results.to_csv(output, index=False)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, help="Query to search for")
    parser.add_argument("--api_key", type=str, help="News API key")
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path. If not provided, will not write to file",
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        default="publishedAt",
        help="Sort by parameter. Default is publishedAt",
    )
    parser.add_argument(
        "--search_in",
        type=str,
        default="title",
        help="Search in parameter. Default is title",
    )
    parser.add_argument(
        "--from_date",
        type=str,
        help="From date in format YYYY-MM-DD",
    )
    parser.add_argument(
        "--to_date",
        type=str,
        help="To date in format YYYY-MM-DD",
    )
    parser.add_argument(
        "--sources",
        type=str,
        help="Comma-separated list of sources to search in",
    )
    parser.add_argument(
        "--exclude_domains",
        type=str,
        help="Comma-separated list of domains to exclude",
    )
    parser.add_argument(
        "--nb_pages",
        type=int,
        default=1,
        help="Number of pages of 100 results to query. Default is 1",
    )
    args = parser.parse_args()

    query_articles(
        args.query,
        args.api_key or os.getenv("NEWS_API_KEY"),
        output=args.output,
        sort_by=args.sort_by,
        search_in=args.search_in,
        from_date=args.from_date,
        to_date=args.to_date,
        sources=args.sources,
        nb_pages=args.nb_pages,
    )
