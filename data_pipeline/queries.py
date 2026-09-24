from pathlib import Path
import sqlite3
import pandas as pd

DB=Path(__file__).resolve().parent/"zepto_books.db"
QUERIES={
"select_where":"SELECT title, price_gbp, rating FROM books WHERE rating >= 4;",
"order_by":"SELECT title, price_inr FROM books ORDER BY price_inr DESC LIMIT 10;",
"distinct":"SELECT DISTINCT category_name FROM categories ORDER BY category_name;",
"between":"SELECT title, price_gbp FROM books WHERE price_gbp BETWEEN 10 AND 30 ORDER BY price_gbp;",
"join_top":"""SELECT c.category_name, b.title, b.rating, b.price_inr
FROM books b JOIN categories c ON b.category_id=c.category_id
ORDER BY b.rating DESC, b.price_inr DESC LIMIT 10;"""
}

def main():
    con=sqlite3.connect(DB)
    outputs={}
    for name,q in QUERIES.items():
        print(f"\n--- {name} ---\n{q}")
        df=pd.read_sql_query(q,con); outputs[name]=df; print(df.to_string(index=False))
    q1=pd.read_sql_query(QUERIES["join_top"],con)
    books=pd.read_sql_query("SELECT * FROM books",con)
    cats=pd.read_sql_query("SELECT * FROM categories",con)
    merged=pd.merge(books,cats,on="category_id")
    merged=merged.sort_values(["rating","price_inr"],ascending=[False,False]).head(10)
    merged=merged[["category_name","title","rating","price_inr"]].reset_index(drop=True)
    print("\n--- JOIN via pd.read_sql ---\n",q1.to_string(index=False))
    print("\n--- Equivalent JOIN via pd.merge ---\n",merged.to_string(index=False))
    print("\nEquivalent:", q1.reset_index(drop=True).equals(merged))
    con.close()

if __name__=="__main__": main()
