import json
import os

from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel

app = FastAPI()

# Initialisation du client OpenAI avec la clé injectée via les variables d'environnement
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class NewsRequest(BaseModel):
    title: str
    summary: str


@app.get("/health")
def health():
    return {"ok": True, "schema": "impacts_v1"}


@app.post("/extract_tickers")
def extract(news: NewsRequest):
    system_prompt = (
        "Tu es un expert financier quantitatif. On te donne une news "
        "(titre + résumé). Identifie les tickers boursiers américains "
        "(actions ou ETFs) impactés et évalue le sentiment pour CHACUN. "
        "Score de -1.0 (très négatif/bearish) à +1.0 (très positif/bullish). "
        "0.0 est neutre. Macro-économie (guerre, taux, inflation) -> utilise "
        "les ETFs (SPY, QQQ, USO, GLD, VXX, TLT, XLE, XLF, XLK, SMH, etc.). "
        'Réponds STRICTEMENT en JSON avec ce format exact : '
        '{"impacts": [{"ticker": "USO", "score": 0.8}, '
        '{"ticker": "AAPL", "score": -0.5}]}. '
        'Si aucun impact sur le marché US, renvoie {"impacts": []}.'
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Titre: {news.title}\nRésumé: {news.summary}"},
            ],
            temperature=0.0,
            timeout=30,
        )
        content = response.choices[0].message.content.strip()
        data = json.loads(content)
        impacts = data.get("impacts", [])
        if not isinstance(impacts, list):
            impacts = []
        return {"impacts": impacts, "raw_response": content}

    except Exception as e:
        print(f"Erreur OpenAI: {e}")
        return {"impacts": [], "raw_response": str(e)}
