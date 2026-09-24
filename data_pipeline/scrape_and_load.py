from __future__ import annotations
import re, sqlite3
from pathlib import Path
import requests
import pandas as pd
from bs4 import BeautifulSoup

BASE = "https://books.toscrape.com"
RATE = 105.50
ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "zepto_books.db"
CSV_PATH = ROOT / "books_clean.csv"

RATING_MAP = {"One":1,"Two":2,"Three":3,"Four":4,"Five":5}

def scrape_books(min_rows=60):
    rows=[]
    page=1
    session=requests.Session()
    while len(rows) < min_rows:
        url=f"{BASE}/catalogue/page-{page}.html"
        r=session.get(url, timeout=20)
        r.raise_for_status()
        soup=BeautifulSoup(r.text, "html.parser")
        cards=soup.select("article.product_pod")
        if not cards: break
        for card in cards:
            title=card.h3.a.get("title", "").strip()
            price_text=card.select_one(".price_color").get_text(strip=True)
            availability=card.select_one(".availability").get_text(" ", strip=True)
            rating_class=next((c for c in card.get("class",[]) if c in RATING_MAP), None)
            rating_text=card.select_one(".star-rating").get("class", ["","-"])[1]
            # category is obtained from the detail page, not inferred from the catalogue.
            detail_url=BASE + "/catalogue/" + card.h3.a["href"].replace("../", "")
            detail=session.get(detail_url, timeout=20); detail.raise_for_status()
            ds=BeautifulSoup(detail.text, "html.parser")
            crumbs=[x.get_text(" ", strip=True) for x in ds.select("ul.breadcrumb li a")]
            category=crumbs[-1] if crumbs else "Unknown"
            rows.append({"title":title,"price":price_text,"star_rating":rating_text,
                         "availability":availability,"category":category})
            if len(rows)>=min_rows: break
        page += 1
        if page > 10: break
    return pd.DataFrame(rows)

def clean_books(df):
    out=df.copy()
    out["price_gbp"]=pd.to_numeric(out["price"].str.replace(r"[^0-9.]", "", regex=True), errors="coerce")
    out["rating"]=out["star_rating"].map(RATING_MAP)
    out["in_stock"]=out["availability"].str.contains("In stock", case=False, na=False)
    # Numeric parse failures are median-imputed as required; categorical failures are dropped.
    for c in ["price_gbp","rating"]:
        out[c]=out[c].fillna(out[c].median())
    out=out.dropna(subset=["title","category"]).copy()
    out["price_inr"]=out["price_gbp"] * RATE
    out["rating"]=out["rating"].round().astype(int).clip(1,5)
    out["in_stock"]=out["in_stock"].astype(bool)
    return out[["title","price_gbp","price_inr","rating","in_stock","category"]]

def create_db(df):
    if DB_PATH.exists(): DB_PATH.unlink()
    con=sqlite3.connect(DB_PATH)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript("""
    CREATE TABLE categories(category_id INTEGER PRIMARY KEY, category_name TEXT UNIQUE NOT NULL);
    CREATE TABLE books(
      book_id INTEGER PRIMARY KEY, title TEXT NOT NULL, price_gbp REAL NOT NULL,
      price_inr REAL NOT NULL, rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
      in_stock INTEGER NOT NULL CHECK(in_stock IN (0,1)),
      category_id INTEGER NOT NULL REFERENCES categories(category_id)
    );
    """)
    cats=sorted(df["category"].unique())
    con.executemany("INSERT INTO categories(category_name) VALUES (?)", [(c,) for c in cats])
    cmap={r[1]:r[0] for r in con.execute("SELECT category_id, category_name FROM categories")}
    records=[]
    for _,r in df.iterrows():
        records.append((r.title,float(r.price_gbp),float(r.price_inr),int(r.rating),int(r.in_stock),cmap[r.category]))
    con.executemany("INSERT INTO books(title,price_gbp,price_inr,rating,in_stock,category_id) VALUES (?,?,?,?,?,?)", records)
    con.commit(); con.close()

def main():
    raw=scrape_books(60)
    clean=clean_books(raw)
    if len(clean)<60 or clean.category.nunique()<3:
        raise RuntimeError(f"Acceptance criteria not met: {len(clean)} rows, {clean.category.nunique()} categories")
    clean.to_csv(CSV_PATH,index=False)
    create_db(clean)
    print(f"Scraped {len(clean)} books across {clean.category.nunique()} categories")
    print(f"Saved {CSV_PATH} and {DB_PATH}; GBP->INR fixed rate={RATE}")

if __name__ == "__main__": main()
