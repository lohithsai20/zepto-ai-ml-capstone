# Data Pipeline

The pipeline scrapes `books.toscrape.com` using `requests` and `BeautifulSoup` until at least 60 books across at least three categories are collected. It cleans price/rating/availability, converts GBP to INR using the required fixed project rate **1 GBP = 105.50 INR**, and loads a normalized SQLite database.

## Run
```bash
python data_pipeline/scrape_and_load.py
python data_pipeline/queries.py
```

Numeric parse failures for price/rating are median-imputed. Rows missing required title/category are dropped because they cannot be represented reliably in the relational model. `in_stock` is parsed from the availability text and stored as SQLite integer 0/1.

The database has `categories(category_id PK, category_name UNIQUE)` and `books(book_id PK, ..., category_id FK)`. The query script demonstrates SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, BETWEEN and a JOIN, then reproduces the JOIN result with `pd.merge`.
