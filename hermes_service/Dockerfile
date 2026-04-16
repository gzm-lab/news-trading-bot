FROM python:3.10-slim

WORKDIR /app

# Installation des dépendances minimales pour l'API et OpenAI
RUN pip install --no-cache-dir fastapi uvicorn openai pydantic httpx requests

# Copie du script du service
COPY main.py /app/main.py

# On expose le port 8000 pour que le bot Python puisse communiquer avec Hermes
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
