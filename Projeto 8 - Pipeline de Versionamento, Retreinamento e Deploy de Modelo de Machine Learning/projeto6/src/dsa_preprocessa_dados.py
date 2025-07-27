# Módulo de Pré-Processamento de Dados

# Importa a biblioteca pandas para manipulação de dados
import pandas as pd  

# Importa a biblioteca numpy para operações numéricas
import numpy as np  

# Importa a função train_test_split para dividir os dados
from sklearn.model_selection import train_test_split  

# Função para gerar dados
def dsa_gera_dados():

    # Define a semente para o gerador de números aleatórios
    np.random.seed(42)

    # Define o tamanho do conjunto de dados
    data_size = 1000

    # Gera dados aleatórios para a variável X1 com distribuição normal
    X1 = np.random.normal(0, 1, data_size)

    # Gera dados aleatórios para a variável X2 com distribuição normal
    X2 = np.random.normal(5, 2, data_size)

    # Gera dados aleatórios para a variável X3 com valores inteiros 0 ou 1
    X3 = np.random.randint(0, 2, data_size)

    # Gera a variável y baseada em uma condição envolvendo X1, X2 e X3
    y = (2*X1 - 3*X2 + X3 > 0).astype(int)

    # Cria um DataFrame com as variáveis geradas
    df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3, 'y': y})

    # Salva o DataFrame em um arquivo CSV
    df.to_csv('dados/originais/dataset.csv', index = False)

# Função para processar dados
def dsa_processa_dados():
    
    # Lê os dados do arquivo CSV
    df = pd.read_csv('dados/originais/dataset.csv')
    
    # Separa as variáveis independentes (X) da variável dependente (y)
    X = df.drop('y', axis = 1)
    y = df['y']
    
    # Divide os dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    
    # Combina os dados de treino em um único DataFrame
    train_data = pd.concat([X_train, y_train], axis = 1)
    
    # Combina os dados de teste em um único DataFrame
    test_data = pd.concat([X_test, y_test], axis = 1)
    
    # Salva os dados de treino em um arquivo CSV
    train_data.to_csv('dados/processados/dados_treino.csv', index = False)
    
    # Salva os dados de teste em um arquivo CSV
    test_data.to_csv('dados/processados/dados_teste.csv', index = False)

# Executa as funções se o script for executado diretamente
if __name__ == "__main__":
    dsa_gera_dados()
    dsa_processa_dados()
    print('\nMódulo de Pré-Processamento de Dados Executado Com Sucesso!\n')




