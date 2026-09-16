import os
import re
import json
from datetime import datetime, timedelta
from newsapi import NewsApiClient


def load_api_key():
    key = os.getenv("NEWS_API_KEY")
    if key:
        return key

    if os.path.exists("news_api.txt"):
        with open("news_api.txt", "r", encoding="utf-8") as f:
            text = f.read()

        if "API Key:" in text:
            key = text.split("API Key:", 1)[1].strip().splitlines()[0].strip()
            if key and "PASTE_YOUR" not in key:
                return key

    raise ValueError("Add your NewsAPI key to news_api.txt or NEWS_API_KEY.")


def fetch_by_day(query: str, start: str, end: str):
    """
    Fetch articles for each calendar day and save them into one JSON file.
    """
    newsapi = NewsApiClient(api_key=load_api_key())
    all_articles = []
    seen_urls = set()

    curr = datetime.fromisoformat(start)
    last = datetime.fromisoformat(end)
    day_count = 0

    while curr <= last:
        day_str = curr.strftime("%Y-%m-%d")
        page = 1
        day_articles = []

        while True:
            res = newsapi.get_everything(
                q=query,
                from_param=day_str,
                to=day_str,
                language="en",
                sort_by="publishedAt",
                page_size=100,
                page=page,
            )

            arts = res.get("articles", [])
            total = res.get("totalResults", 0)

            for article in arts:
                url = article.get("url")
                if url and url in seen_urls:
                    continue
                if url:
                    seen_urls.add(url)
                day_articles.append(article)

            if len(arts) < 100 or page * 100 >= total:
                break

            page += 1

        print(f"{day_str}: fetched {len(day_articles)} articles")
        all_articles.extend(day_articles)
        curr += timedelta(days=1)
        day_count += 1

    safe_query = re.sub(r"[^A-Za-z0-9]+", "_", query).strip("_")
    out_file = f"news_{safe_query}_{start}_{end}.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"articles": all_articles}, f, ensure_ascii=False, indent=2)

    print(
        f"\nSaved {len(all_articles)} total articles "
        f"({day_count} days) to {out_file}"
    )


if __name__ == "__main__":
    fetch_by_day('AAPL OR "Apple Inc"', "2026-08-16", "2026-09-16")
