"""text2sparql-api"""

import fastapi

app = fastapi.FastAPI(
    title="TEXT2SPARQL API Example",
)

KNOWN_DATASETS = [    
    "https://text2sparql.aksw.org/2025/dbpedia/",
]

@app.get("/")
async def get_answer(question: str, dataset: str):
    
    if dataset not in KNOWN_DATASETS:
            
        raise fastapi.HTTPException(404, "Unknown dataset ...")
        
    return {            
        "dataset": dataset,
        "question": question,
        "query": "... SPARQL here ..."
    }
