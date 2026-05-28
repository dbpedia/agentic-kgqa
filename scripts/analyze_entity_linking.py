import yaml
import re
import sys
import pandas as pd
sys.path.insert(0, '/Users/siddharth/Desktop/DBpedia/agentic-kgqa')
from src.entity_linking import RedisEntityLinking

with open('benchmark/questions_db25.yaml', 'r') as f:
    data = yaml.safe_load(f)

def extract_entities(sparql):
    return re.findall(r'<http://dbpedia\.org/resource/([^>]+)>', sparql)

def uri_to_surface(uri):
    # Convert DBpedia URI to natural surface form
    # e.g. "Keanu_Reeves" -> "Keanu Reeves"
    return uri.replace('_', ' ').replace('(', '').replace(')', '')

el = RedisEntityLinking()

results = []
for q in data['questions']:
    qid = q['id']
    question = q['question']['en']
    sparql = q['query']['sparql']
    gold_entities = extract_entities(sparql)
    
    for entity_uri in gold_entities:
        surface = uri_to_surface(entity_uri)
        redis_result = el.lookup(surface, top_k=3)
        redis_entities = redis_result.index.tolist() if len(redis_result) > 0 else []
        
        # Check if correct entity is in top results
        found_correct = any(entity_uri in r for r in redis_entities)
        
        results.append({
            'id': qid,
            'question': question[:60],
            'gold_uri': entity_uri,
            'surface_form': surface,
            'redis_top1': redis_entities[0] if redis_entities else 'NO RESULT',
            'found_correct': found_correct,
            'redis_score': redis_result.iloc[0]['score'] if len(redis_result) > 0 else 0
        })
        
        print(f"Q{qid} | '{surface}' -> {redis_entities[0] if redis_entities else 'NO RESULT'} | Correct: {found_correct}")

df = pd.DataFrame(results)
df.to_csv('entity_linking_analysis.csv', index=False)
print(f"\nTotal entities tested: {len(df)}")
print(f"Correct top-1: {df['found_correct'].sum()} ({df['found_correct'].mean()*100:.1f}%)")
print(f"No result: {(df['redis_top1']=='NO RESULT').sum()}")