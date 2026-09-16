import os
import json
from datetime import datetime, timedelta
from newsapi import NewsApiClient

API_KEY = ""  

def fetch_by_day(ticker: str, start: str, end: str):
    """
    Fetch up to 100 articles per calendar day between start/end (inclusive),
    and write them all into one JSON file.
    """
    newsapi = NewsApiClient(api_key=API_KEY)
    all_articles = []

    curr = datetime.fromisoformat(start)
    last = datetime.fromisoformat(end)
    day_count = 0

    while curr <= last:
        day_str = curr.strftime("%Y-%m-%d")
        res = newsapi.get_everything(
            q          = ticker,
            from_param = day_str,
            to         = day_str,
            language   = "en",
            sort_by    = "publishedAt",
            page_size  = 100,
        )
        arts = res.get("articles", [])
        print(f"{day_str}: fetched {len(arts)} articles")
        all_articles.extend([
            {**a, "publishedAt": day_str}  # day-only
            for a in arts
        ])
        curr += timedelta(days=1)
        day_count += 1

    out_file = f"news_{ticker.replace(' ','_')}_{start}_{end}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"articles": all_articles}, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(all_articles)} total articles "
          f"({day_count} days) to {out_file}")

if __name__ == "__main__":
    fetch_by_day("AAPL OR Apple Inc", "2025-07-01", "2025-07-24")
