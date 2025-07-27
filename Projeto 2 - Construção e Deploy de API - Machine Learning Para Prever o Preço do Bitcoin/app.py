# -*- coding: utf-8 -*-
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
from indicadores import dsa_calcula_indicadores
import yfinance as yf

tags_metadata = [{
    "name": "DSA-Projeto2",
    "description": "Prevendo o Preço de Bitcoin com Machine Learning"
}]

app = FastAPI(
    title="Bitcoin Price API",
    description="DSA - Projeto2",
    version="1.0",
    contact={"name": "DSA", "url": "https://www.datascienceacademy.com.br"},
    openapi_tags=tags_metadata
)

class Features(BaseModel):
    Model: str

@app.get("/")
def message():
    return "Esta é uma API Para Prever o Preço de Bitcoin com Machine Learning. Use o Método Adequado."

@app.post("/predict", tags=["DSA-Projeto2"])
async def predict(entrada: Features):
    btc_ticker = yf.Ticker("BTC-USD")
    valor_historico_btc = btc_ticker.history(period="200d", actions=False)
    valor_historico_btc = valor_historico_btc.tz_localize(None)
    valor_historico_btc = dsa_calcula_indicadores(valor_historico_btc)
    valor_historico_btc = valor_historico_btc.sort_index(ascending=False)

    dados_entrada = valor_historico_btc.iloc[0, :].fillna(0).array
    dados_entrada = dados_entrada.reshape(1, -1)

    scaler = load("scaler_dsa.bin")
    dados_entrada = scaler.transform(dados_entrada)

    Model = entrada.Model

    if Model == "Machine Learning":
        model = load("modelo_dsa.joblib")
    else:
        return {"erro": "Modelo não reconhecido."}

    previsao = model.predict(dados_entrada)
    ultimo_preco = valor_historico_btc.iloc[0, 3]

    return {
        "Modelo": Model,
        "Último Preço": round(ultimo_preco, 2),
        "Previsão Para o Próximo Dia": round(previsao.tolist()[0], 2)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)
