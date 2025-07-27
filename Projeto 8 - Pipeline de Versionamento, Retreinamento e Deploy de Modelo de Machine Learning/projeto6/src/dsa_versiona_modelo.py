# Módulo de Versionamento do Modelo

# Importa a biblioteca os para manipulação de arquivos e diretórios
import os

# Importa a biblioteca joblib para salvar e carregar modelos
import joblib

# Função para salvar uma nova versão do modelo
def dsa_salva_nova_versao_modelo(model, version):
    
    # Define o caminho do arquivo para salvar o modelo
    model_path = f'modelos/modelo_dsa_v{version}.pkl'
    
    # Salva o modelo no caminho especificado
    joblib.dump(model, model_path)

# Função para listar as versões dos modelos
def dsa_lista_versao_modelos():
    
    # Define o diretório onde os modelos estão armazenados
    models_dir = 'modelos/'
    
    # Lista todos os arquivos no diretório que começam com 'modelo_dsa_v'
    models = [f for f in os.listdir(models_dir) if f.startswith('modelo_dsa_v')]
    
    # Retorna a lista de modelos encontrados
    return models

# Executa o código se o script for executado diretamente
if __name__ == "__main__":
    
    # Obtém a lista de versões dos modelos
    models = dsa_lista_versao_modelos()
    
    # Imprime as versões disponíveis
    print("\nVersões Disponíveis:\n")
    
    # Itera sobre a lista de modelos e imprime cada um
    for model in models:
        print(model)

    print('\nMódulo de Versionamento do Modelo Executado Com Sucesso!\n')



    