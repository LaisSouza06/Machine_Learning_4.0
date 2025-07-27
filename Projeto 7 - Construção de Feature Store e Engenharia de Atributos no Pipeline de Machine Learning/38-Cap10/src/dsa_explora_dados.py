# Projeto 5 - Construção de Feature Store e Aplicação de Engenharia de Atributos 
# Módulo de Exploração e Visualização dos Dados

# Imports
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_feature_distributions(df_features_dsa):
    
    # Calcula o número de características numéricas, excluindo a coluna de índice ou target
    num_features = len(df_features_dsa.columns) - 1  
    
    # Define o número de colunas para os subplots
    cols = 3  
    
    # Calcula o número necessário de linhas para os subplots baseado no número de características
    rows = (num_features + cols - 1) // cols  

    # Cria uma figura com um grid de subplots com o número determinado de linhas e colunas
    fig, axes = plt.subplots(nrows = rows, ncols = cols, figsize = (15, rows * 3))
    
    # Transforma a matriz de eixos em um array unidimensional para fácil indexação
    axes = axes.flatten()

    # Loop que percorre cada coluna da DataFrame (excluindo a última, a variável alvo)
    for i, col in enumerate(df_features_dsa.columns[:-1]):  
        
        # Cria um histograma da coluna com uma linha de densidade
        sns.histplot(df_features_dsa[col], ax = axes[i], kde = True)
        
        # Define o título do subplot como o nome da coluna
        axes[i].set_title(col)
        
        # Define o nome do eixo Y
        axes[i].set_ylabel('Count')

    # Desativa os eixos que não são usados (caso haja menos características que subplots)
    for i in range(num_features, len(axes)):
        axes[i].set_visible(False)

    # Ajusta os subplots para evitar sobreposição e otimizar o layout
    plt.tight_layout()
    
    # Exibe a figura
    plt.show()


def plot_feature_correlations(df_features_dsa):
    
    # Calcula a matriz de correlação das características
    correlation_matrix = df_features_dsa.corr()
    
    # Cria uma figura com tamanho especificado
    plt.figure(figsize = (8, 6))
    
    # Plota a matriz de correlação com anotações e um mapa de cores
    sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm')
    
    # Define o título do gráfico
    plt.title('Feature Correlations')
    
    # Exibe o gráfico
    plt.show()

def analisa_dados(df_features_dsa):
    
    # Imprime um título para a distribuição das características
    print("\nGráfico de Feature Distributions.")
    
    # Chama a função para plotar as distribuições
    plot_feature_distributions(df_features_dsa)

    # Imprime um título para as correlações das características
    print("\nGráfico de Feature Correlations.")
    
    # Chama a função para plotar as correlações
    plot_feature_correlations(df_features_dsa)



