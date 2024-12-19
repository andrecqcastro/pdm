from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from transformers import pipeline

# Carregar modelo de análise de sentimento
model_name = "EdwardSJ151/bert-amazon-reviews"
sentiment_model = pipeline("text-classification", model=model_name)

# Inicializar o aplicativo FastAPI
app = FastAPI()

# Definir o esquema de entrada
class SentimentInput(BaseModel):
    texts: List[str]  # Lista de textos para análise

# Definir o esquema de saída
class SentimentOutput(BaseModel):
    label: str
    score: float

@app.post("/sentiment-analysis", response_model=List[SentimentOutput])
async def analyze_sentiments(input_data: SentimentInput):
    try:
        # Fazer inferência para cada texto da entrada
        predictions = [
            {
                "label": sentiment_model(text)[0]["label"],
                "score": sentiment_model(text)[0]["score"]
            }
            for text in input_data.texts
        ]

        return predictions

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro na predição: {str(e)}")
