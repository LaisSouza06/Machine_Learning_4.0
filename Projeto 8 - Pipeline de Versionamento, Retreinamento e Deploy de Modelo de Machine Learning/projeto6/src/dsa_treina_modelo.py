# Módulo de Treinamento do Modelo

# Importa a biblioteca joblib para salvar e carregar modelos
import joblib  

# Importa a biblioteca pandas para manipulação de dados
import pandas as pd  

 # Importa o classificador RandomForest da biblioteca sklearn
from sklearn.ensemble import RandomForestClassifier 

# Função para treinar o modelo
def dsa_treina_modelo():

    # Lê os dados de treino do arquivo CSV
    dados_treino = pd.read_csv('dados/processados/dados_treino.csv')
    
    # Separa as variáveis independentes (X_treino) da variável dependente (y_treino)
    X_treino = dados_treino.drop('y', axis = 1)
    y_treino = dados_treino['y']
    
    # Cria uma instância do modelo RandomForestClassifier
    modelo_dsa = RandomForestClassifier(n_estimators = 100, random_state = 42)
    
    # Treina o modelo com os dados de treino
    modelo_dsa.fit(X_treino, y_treino)
    
    # Salva o modelo treinado em um arquivo
    joblib.dump(modelo_dsa, 'modelos/modelo_dsa_v1.pkl')

# Executa a função se o script for executado diretamente
if __name__ == "__main__":
    dsa_treina_modelo()
    print('\nMódulo de Treinamento do Modelo Executado Com Sucesso!\n')
