# Projeto 1 - Construção e Deploy de Modelo de Machine Learning

# Implementação do Software de Deploy do Modelo

# Imports
from flask import Flask, render_template, request, jsonify
import joblib
import warnings
warnings.filterwarnings('ignore')

# App
app = Flask(__name__)

# Carregar o modelo e transformadores do disco
modelo_dsa = joblib.load('modelos/modelo_logistica.pkl')
le_tipo_embalagem = joblib.load('modelos/transformador_tipo_embalagem.pkl')
le_tipo_produto = joblib.load('modelos/transformador_tipo_produto.pkl')

# Define a rota principal para a página inicial e aceita apenas requisições GET
@app.route('/', methods = ['GET'])
def index():

    # Renderiza a página inicial usando o template.html
    return render_template('template.html')

# Define uma rota para fazer previsões e aceita apenas requisições POST
@app.route('/predict', methods = ['POST'])
def predict():

    # Extrai o valor de 'Peso' do formulário enviado
    peso = int(request.form['Peso'])
    
    # Transforma o tipo de embalagem usando o label encoder previamente ajustado
    tipo_embalagem = le_tipo_embalagem.transform([request.form['tipo_embalagem']])[0]
    
    # Usa o modelo para fazer uma previsão com base no peso e tipo de embalagem
    prediction = modelo_dsa.predict([[peso, tipo_embalagem]])[0]
    
    # Converte a previsão codificada de volta ao seu rótulo original
    tipo_produto = le_tipo_produto.inverse_transform([prediction])[0]
    
    # Renderiza a página inicial com a previsão incluída
    return render_template('template.html', prediction = tipo_produto)


# App
if __name__ == '__main__':
    app.run()





    

