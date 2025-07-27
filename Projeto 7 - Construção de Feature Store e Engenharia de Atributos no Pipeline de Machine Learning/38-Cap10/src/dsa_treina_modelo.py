# Projeto 5 - Construção de Feature Store e Aplicação de Engenharia de Atributos 
# Módulo de Treinamento do Modelo de Machine Learning

# Imports
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def treina_avalia_modelo(df_features_dsa):

    # Divide os dados em conjuntos de treino e teste, usando 80% para treino e 20% para teste, fixando a aleatoriedade para reprodutibilidade
    X_treino, X_teste, y_treino, y_teste = train_test_split(df_features_dsa.iloc[:, :-1], df_features_dsa['target'], test_size = 0.2, random_state = 42)
    
    # Cria um modelo de classificador de florestas aleatórias com 100 árvores, fixando a aleatoriedade para reprodutibilidade
    modelo_dsa = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Treina o modelo de florestas aleatórias usando os dados de treino
    modelo_dsa.fit(X_treino, y_treino)

    # Utiliza o modelo treinado para fazer previsões sobre o conjunto de teste
    previsoes = modelo_dsa.predict(X_teste)
    
    # Calcula a acurácia comparando as previsões com os verdadeiros valores de teste
    acuracia = accuracy_score(y_teste, previsoes)
    
    # Gera um relatório de classificação, que inclui precisão, recall e F1-score para cada classe
    class_report = classification_report(y_teste, previsoes)

    # Retorna o modelo treinado, os dados de teste, os valores verdadeiros de teste, as previsões, a acurácia e o relatório de classificação
    return modelo_dsa, X_teste, y_teste, previsoes, acuracia, class_report


