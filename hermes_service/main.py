from fastapi import FastAPI
from pydantic import BaseModel
import os
import json
from openai import OpenAI

app = FastAPI()

# Initialisation du client OpenAI avec la clé injectée via les variables d'environnement
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class NewsRequest(BaseModel):
    title: str
    summary: str

@app.post("/extract_tickers")
def extract(news: NewsRequest):
    system_prompt = (
        "Tu es un expert financier quantitatif. On te donne une news (titre + résumé). "
        "Identifie les tickers boursiers américains (actions ou ETFs) impactés et évalue le sentiment pour CHACUN. "
        "Score de -1.0 (très négatif/bearish) à +1.0 (très positif/bullish). 0.0 est neutre. "
        "Macro-économie (guerre, taux, inflation) -> utilise les ETFs (SPY, USO, GLD, VXX, TLT, etc.). "
        "Réponds STRICTEMENT en JSON avec ce format exact : {\"impacts\": [{\"ticker\": \"USO\", \"score\": 0.8}, {\"ticker\": \"AAPL\", \"score\": -0.5}]}. "
        "Si aucun impact sur le marché US, renvoie {\"impacts\": []}."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Titre: {news.title}\nRésumé: {news.summary}"}
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        data = json.loads(content)
        
        return {"impacts": data.get("impacts", []), "raw_response": content}
        
    except Exception as e:
        print(f"Erreur OpenAI: {e}")
        return {"impacts": [], "raw_response": str(e)}
