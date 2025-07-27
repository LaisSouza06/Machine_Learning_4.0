# Projeto 3 - Deploy de Modelo de Machine Learning na Nuvem AWS Para Gestão de Escolas
# App Web

# Imports
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

# Cria a app
app = Flask(__name__)

# Função para carregar o modelo ou scaler
def carregar_arquivo(caminho):
    with open(caminho, 'rb') as file:
        return pickle.load(file)

# Carrega o modelo e o scaler
modelo = carregar_arquivo('modelo_dsa_final.pkl')
scaler = carregar_arquivo('scaler_final.pkl')

# Rota para a página web de entrada
@app.route('/')
def index():
    return render_template('index.html')

# Rota para a função de previsão
@app.route('/predict', methods=['POST'])
def previsao():
    
    try:

        # Extrai os valores enviados via formulário
        exame_ingles = float(request.form.get('exame_ingles', 0))
        exame_psico = int(request.form.get('exame_psico', 0))
        nota_qi = int(request.form.get('nota_qi', 0))

        # Validação de entrada
        if not (0 <= exame_ingles <= 10 and 0 <= nota_qi <= 200 and 0 <= exame_psico <= 100):
            raise ValueError("Valores de entrada inválidos")

        # Cria um array com os dados de entrada ajustando o shape
        dados_entrada = np.array([exame_ingles, nota_qi, exame_psico]).reshape(1, 3)

        # Dataframe com dados e nomes de colunas
        dados_entrada_df = pd.DataFrame(dados_entrada, columns=['nota_exame_ingles', 'valor_qi', 'nota_exame_psicotecnico'])

        # Padroniza os dados de entrada
        dados_entrada_padronizados = scaler.transform(dados_entrada_df)

        # Faz a previsão com o modelo usando os dados padronizados (da mesma forma que o modelo foi treinado)
        pred = modelo.predict(dados_entrada_padronizados)
        
        resultado = 'O Aluno Poderá Ser Inscrito no Curso' if pred[0] == 1 else 'O Aluno Não Poderá Ser Inscrito no Curso'

    except Exception as e:
        resultado = f"Erro na previsão: {e}"

    return render_template('index.html', result=resultado)

if __name__ == "__main__":
    app.run()
