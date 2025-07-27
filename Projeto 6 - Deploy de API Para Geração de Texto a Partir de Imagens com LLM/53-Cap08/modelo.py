# Projeto 4 - Deploy de API Para Geração de Texto a Partir de Imagens com LLM
# Módulo para download do LLM

# Importa a classe ViltProcessor para processamento de imagem e texto, e ViltForQuestionAnswering para o modelo de QA
from transformers import ViltProcessor, ViltForQuestionAnswering

# Importa a biblioteca PIL para manipulação de imagens
from PIL import Image

# Carrega o processador pré-treinado específico para tarefas de QA visuais
dsa_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Carrega o modelo pré-treinado para responder a perguntas baseadas em imagens e texto
dsa_model = ViltForQuestionAnswering.from_pretrained('dandelin/vilt-b32-finetuned-vqa')

# Define uma função pipeline para processar texto e imagem e obter uma resposta
def dsa_model_pipeline(text:str, image:Image):

    # Processa a imagem e o texto juntos, preparando-os para o modelo
    encoding = dsa_processor(image, text, return_tensors = "pt")

    # Passa os dados processados pelo modelo e obtém a saída
    outputs = dsa_model(**encoding)

    # Extrai os logits (pontuações não normalizadas) da saída do modelo
    logits = outputs.logits
    
    # Encontra o índice da maior pontuação nos logits, indicando a resposta mais provável
    index = logits.argmax(-1).item()

    # Retorna a etiqueta associada ao índice de maior pontuação como a resposta
    return dsa_model.config.id2label[index]
