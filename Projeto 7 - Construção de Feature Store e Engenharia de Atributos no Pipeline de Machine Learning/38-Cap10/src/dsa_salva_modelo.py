# Projeto 5 - Construção de Feature Store e Aplicação de Engenharia de Atributos 
# Módulo Para Salvar o Modelo e as Previsões

# Imports
import json
import pandas as pd
from joblib import dump

def salva_modelo(model, model_path):

    # Salva o modelo treinado no caminho especificado usando a função 'dump' da biblioteca joblib
    dump(model, model_path)

def salva_previsoes(predictions, y_test, predictions_path):

    # Cria um DataFrame a partir das previsões e dos valores reais de teste
    predictions_df = pd.DataFrame({'ValorReal': y_test, 'ValorPrevisto': predictions})
    
    # Salva o DataFrame em um arquivo CSV no caminho especificado sem incluir o índice
    predictions_df.to_csv(predictions_path, index = False)

def salva_info(run_info, run_info_path):

    # Abre um arquivo no modo de escrita no caminho especificado
    with open(run_info_path, 'w') as f:
        
        # Salva as informações da execução (run_info) no arquivo usando o formato JSON
        json.dump(run_info, f)

