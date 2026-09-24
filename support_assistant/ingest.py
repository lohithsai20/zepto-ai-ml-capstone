from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

ROOT=Path(__file__).resolve().parent
DOCS=ROOT/'docs'
DB=ROOT/'chroma_db'
COLLECTION='zepto_policy'
MODEL_NAME='all-MiniLM-L6-v2'

def build_collection():
    client=chromadb.PersistentClient(path=str(DB))
    try: client.delete_collection(COLLECTION)
    except Exception: pass
    col=client.get_or_create_collection(COLLECTION, metadata={'hnsw:space':'cosine'})
    model=SentenceTransformer(MODEL_NAME)
    ids=[]; docs=[]; metas=[]
    for path in sorted(DOCS.glob('doc_*.txt')):
        text=path.read_text(encoding='utf-8').strip()
        ids.append(f'{path.stem}_chunk_0'); docs.append(text); metas.append({'document_id':path.stem,'chunk_id':f'{path.stem}_chunk_0'})
    embeddings=model.encode(docs,normalize_embeddings=True).tolist()
    col.add(ids=ids,documents=docs,metadatas=metas,embeddings=embeddings)
    print(f'Indexed {len(ids)} chunks into {COLLECTION}')
    return col

if __name__=='__main__': build_collection()
