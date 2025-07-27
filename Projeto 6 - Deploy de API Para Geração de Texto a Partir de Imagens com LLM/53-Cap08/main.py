# Projeto 4 - Deploy de API Para Geração de Texto a Partir de Imagens com LLM
# Módulo da API

# Importa as classes necessárias do FastAPI para criar a API e gerenciar arquivos e formulários
from fastapi import FastAPI, UploadFile, File, Form

# Importa a função do modelo para processar a imagem e o texto e retornar uma resposta
from modelo import dsa_model_pipeline

# Importa a biblioteca PIL para manipulação de imagens
from PIL import Image

# Importa io para manipulação de streams de bytes
import io

# Importa uvicorn, um servidor WSGI, para hospedar nossa aplicação FastAPI
import uvicorn

# Cria uma instância do FastAPI
app = FastAPI()

# Define uma rota raiz que retorna uma mensagem
# Este método define uma rota HTTP GET para a raiz do seu servidor (ou seja, "/"). 
# Quando alguém acessa a URL base da sua API, essa função é chamada e retorna um texto.
# É uma forma simples de verificar se a sua API está funcionando corretamente.
@app.get("/")
def inicio():
    return {"DSA": "Projeto4"}

# Raiz: "http://localhost:3000/"
# API:  "http://localhost:3000/api"

# O async (usado abaixo) antes da definição da função em um aplicativo web construído com um framework como FastAPI 
# indica que a função é assíncrona. Isso significa que a função pode ser pausada e retomada, permitindo que Python execute 
# outras tarefas enquanto espera por uma operação de entrada/saída (I/O) ser concluída, como acessar um banco de dados ou 
# fazer uma requisição de rede. O uso de async permite que o servidor web lide com muitas requisições simultaneamente de forma mais 
# eficiente, sem bloquear o processamento enquanto espera que as tarefas de I/O sejam concluídas.

# Define uma rota para a previsão que utiliza o modelo especificado
# Este método define uma rota HTTP POST para "/api". 
# É usado para fazer previsões com base nas características (features) enviadas no corpo da requisição. 
# A função api recebe objetos como argumento, que contém os valores das características necessárias para fazer a previsão. 
# Dentro dessa função, você vai extrair essas características, fazer algum pré-processamento necessário e então passá-las para 
# o seu modelo de machine learning para obter a previsão. Finalmente, a previsão é retornada como resposta à requisição.
@app.post("/api")
async def api(text: str = Form(...), image: UploadFile = File(...)):

    # Lê o conteúdo da imagem enviada
    image_contents = await image.read()
    
    # Abre a imagem a partir do seu conteúdo binário
    image = Image.open(io.BytesIO(image_contents))

    # Chama a função do modelo, passando o texto e a imagem processada, e armazena o resultado
    resultado = dsa_model_pipeline(text, image)

    # Retorna o resultado processado pelo modelo em um dicionário JSON
    return {"Resposta": resultado}

# Inicia o servidor WSGI Uvicorn com a API 
if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 3000)







