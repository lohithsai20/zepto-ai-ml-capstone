from fastapi import FastAPI
from .graph import graph
from .models import AskRequest, AskResponse

app=FastAPI(title='Zepto Policy Support Assistant')

@app.post('/ask',response_model=AskResponse)
def ask(req: AskRequest):
    state=graph.invoke({'query':req.query})
    return state['response']
