from __future__ import annotations
import os, json
from pathlib import Path
from typing import TypedDict
import chromadb
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, END
from .models import AskResponse
from .prompts import STRUCTURED_PROMPT

ROOT=Path(__file__).resolve().parent; DB=ROOT/'chroma_db'; COLLECTION='zepto_policy'
MODEL_NAME='all-MiniLM-L6-v2'
model=SentenceTransformer(MODEL_NAME)
client=chromadb.PersistentClient(path=str(DB))

def collection():
    try:
        return client.get_collection(COLLECTION)
    except Exception:
        from .ingest import build_collection
        build_collection()
        return client.get_collection(COLLECTION)

def mock_enabled(): return os.getenv('MOCK_LLM','1') != '0'

def real_llm(prompt):
    # Optional extension: Groq-compatible chat endpoint. It is never called in default/mock mode.
    import requests
    key=os.getenv('GROQ_API_KEY')
    if not key: raise RuntimeError('GROQ_API_KEY is required when MOCK_LLM=0')
    r=requests.post('https://api.groq.com/openai/v1/chat/completions',headers={'Authorization':f'Bearer {key}','Content-Type':'application/json'},json={'model':os.getenv('GROQ_MODEL','llama-3.1-8b-instant'),'messages':[{'role':'user','content':prompt}],'temperature':0},timeout=30)
    r.raise_for_status(); return r.json()['choices'][0]['message']['content']

def parse_real(raw, sources):
    try:
        obj=json.loads(raw); return AskResponse(**obj)
    except Exception:
        raise ValueError('Invalid structured output')

def classify_intent(state):
    q=state['query'].lower()
    keywords=['delivery','return','refund','membership','tracking','track','order','cancel','gift card','support hours']
    intent='policy_question' if any(k in q for k in keywords) else 'general_question'
    if not mock_enabled():
        # Optional LLM branch, with deterministic fallback if a model returns an invalid label.
        raw=real_llm(f"Classify as exactly policy_question or general_question. Query: {state['query']}")
        if 'policy_question' in raw: intent='policy_question'
        elif 'general_question' in raw: intent='general_question'
    state['intent']=intent; return state

def retrieve_and_answer(state):
    qemb=model.encode([state['query']],normalize_embeddings=True).tolist()
    res=collection().query(query_embeddings=qemb,n_results=3,include=['documents','metadatas','distances'])
    docs=res['documents'][0]; metas=res['metadatas'][0]; ids=[m['chunk_id'] for m in metas]
    context='\n\n'.join(f"[{m['chunk_id']}] {d}" for m,d in zip(metas,docs))
    if mock_enabled():
        answer=f"Based on the retrieved context: {docs[0][:200]}"
        out=AskResponse(answer=answer,sources=ids,confidence=1.0)
    else:
        prompt=STRUCTURED_PROMPT.format(context=context,question=state['query'])
        last=None
        for attempt in range(3):
            try:
                raw=real_llm(prompt if attempt==0 else prompt+'\nCorrective instruction: return ONLY valid JSON matching the required schema.')
                out=parse_real(raw,ids); break
            except Exception as e: last=e
        else: out=AskResponse(answer=f'ERROR: structured response validation failed: {last}',sources=ids,confidence=0.0)
    state['response']=out; state['retrieved_ids']=ids; return state

def direct_answer(state):
    if mock_enabled(): out=AskResponse(answer='I can only answer questions about Zepto policies right now.',sources=[],confidence=1.0)
    else:
        raw=real_llm(STRUCTURED_PROMPT.format(context='No policy context was retrieved.',question=state['query']))
        try: out=parse_real(raw,[])
        except Exception:
            last=None
            for _ in range(2):
                try: out=parse_real(real_llm(STRUCTURED_PROMPT.format(context='No policy context was retrieved.',question=state['query'])+'\nReturn ONLY valid JSON.'),[]); break
                except Exception as e: last=e
            else: out=AskResponse(answer=f'ERROR: structured response validation failed: {last}',sources=[],confidence=0.0)
    state['response']=out; return state

class GraphState(TypedDict, total=False):
    query: str
    intent: str
    retrieved_ids: list[str]
    response: AskResponse

def route(state): return 'retrieve_and_answer' if state['intent']=='policy_question' else 'direct_answer'

def build_graph():
    g=StateGraph(GraphState)
    g.add_node('classify_intent',classify_intent); g.add_node('retrieve_and_answer',retrieve_and_answer); g.add_node('direct_answer',direct_answer)
    g.set_entry_point('classify_intent')
    g.add_conditional_edges('classify_intent',route,{'retrieve_and_answer':'retrieve_and_answer','direct_answer':'direct_answer'})
    g.add_edge('retrieve_and_answer',END); g.add_edge('direct_answer',END)
    return g.compile()

graph=build_graph()
