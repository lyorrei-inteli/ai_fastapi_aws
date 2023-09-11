# Deploy de modelo de Machine Learning na nuvem
  
## Descrição do Projeto

Este projeto consiste na criação e deploy de um modelo de Machine Learning para prever acidentes em rodovias brasileiras entre os anos de 2010 a 2023. Utilizou-se o dataset "Brazil: Total highway crashes 2010 - 2023" para treinar o modelo e posteriormente oferecer uma API de acesso para realizar predições.

## DockerHub

A imagem Docker deste projeto pode ser encontrada no seguinte link:
- [Link para a imagem Docker no DockerHub](https://hub.docker.com/repository/docker/lyorrei/ai_fastapi_aws/general) (https://hub.docker.com/repository/docker/lyorrei/ai_fastapi_aws/general)

## Ambiente de Desenvolvimento

As seguintes ferramentas foram utilizadas:

- **Python** (versão 3.10.13): Linguagem de programação utilizada para desenvolvimento da API e scripts de pré-processamento e treinamento.
- **FastAPI**: Framework web usado para desenvolver a API.
- **PyCaret**: Framework de Machine Learning utilizado para treinamento e avaliação do modelo.
- **Jupyter Notebook**: Ambiente de desenvolvimento interativo para a construção dos notebooks de ETL e modelagem.

### Como Executar localmente
#### Via normal
1. Acesse o diretório do projeto.
2. Descomprima o arquivo `/data/highway_crashes.zip` para obter os arquivos de dados (git não deixou subir os dois arquivos completos por causa do tamanho).
3. Certifique-se de ter todas as ferramentas mencionadas acima instaladas:

    ``
      pip install -r requirements.txt
    ``

4. Execute o arquivo `app.py` para iniciar a API.

    ``
      python3 app.py
    ``

Caso queira realizar a etapa de ETL novamente, execute o arquivo `/model/etl.ipynb`. Para realizar o treinamento do modelo novamente, execute o arquivo `/model/training.ipynb`.

#### Via Docker
1. Acesse o diretório do projeto.
2. Descomprima o arquivo `/data/highway_crashes.zip` para obter os arquivos de dados (git não deixou subir os dois arquivos completos por causa do tamanho).
3. Certifique-se de ter o Docker instalado.
4. Execute o seguinte comando para obter a imagem:
   ```bash
   docker pull lyorrei/ai_fastapi_aws:v1
   ```
5. Inicie a aplicação com:
   ```
    docker run -d -p 8000:8000 lyorrei/ai_fastapi_aws:v1
   ```

## Estrutura dos Arquivos

- `app.py`: Contém o código da API FastAPI para realizar predições usando o modelo treinado.
- `/model/etl.ipynb`: Jupyter Notebook com o script ETL para extração, transformação e carga dos dados.
- `/model/training.ipynb`: Jupyter Notebook onde ocorre o treinamento do modelo utilizando o PyCaret.
- `/model/final_model.pkl`: Modelo final salvo em disco após o treinamento.
- `/data/highway_crashes.csv`: Arquivo CSV contendo dados brutos de acidentes nas rodovias.
- `/data/highway_crashes_processed.csv`: Arquivo CSV contendo dados de acidentes processados e prontos para treinamento e predição.

## Modelo de Machine Learning

Foi escolhido o modelo "Light Gradient Boosting Machine" (conhecido como lightgbm), do framework PyCaret para classificar os dados de acidentes em rodovias. A escolha deste modelo foi baseada em sua performance superior em relação aos outros modelos testados, tendo uma acurácia de 0.9847 e um F1-score de 0.4294.

## Deploy na AWS

O deploy do modelo foi realizado em uma instância EC2 na AWS. Para executar a aplicação na AWS, a imagem Docker do projeto foi puxada do DockerHub e colocada para rodar na instância.

## Documentação da API

Caso queira acessar a documentação da API, após rodar a api, acesse o link abaixo (swagger):
- [Link para a documentação da API](http://localhost:8000/docs) (http://localhost:8000/docs)


**Endpoint**:
- POST -> `/predict/`: Recebe dados de entrada no formato JSON e retorna a predição do modelo.

**Entrada**:
- JSON com os valores das características. Exemplo:
````
{
    "car": 1.5,
    "bicycle": 0.0,
    "trucks": 2.5,
    "motorbike": 1.2,
    "bus": 0.3,
    "others": 0.0,
    "animals": 0.0,
    "special_cargos": 1.7,
    "tractors": 0.1,
    "utilities": 1.4,
    "unharmed": 0.5,
    "slight_injury": 1.1,
    "moderate_injury": 0.3,
    "serious_injury": 0.9,
    "month": 6,
    "year": 2023,
    "dayofweek": 4,
    "road_info_Descending_Curve": 1,
    "road_info_East_Going": 0,
    "road_info_North_Going": 1,
    "road_info_Other": 0,
    "road_info_South_Going": 0,
    "road_info_Top_Curve": 1,
    "road_info_West_Going": 0,
    "accident_place_Autopista_Fernão_Dias": 0,
    "accident_place_Autopista_Fluminense": 0,
    "accident_place_Autopista_Litoral_Sul": 1,
    "accident_place_Autopista_Planalto_Sul": 0,
    "accident_place_Autopista_Regis_Bittencourt": 0,
    "accident_place_Concebra": 0,
    "accident_place_Concepa": 0,
    "accident_place_Concer": 0,
    "accident_place_Cro": 0,
    "accident_place_Crt": 0,
    "accident_place_ECO050": 0,
    "accident_place_ECO101": 1,
    "accident_place_Ecoponte": 0,
    "accident_place_Ecoriominas": 0,
    "accident_place_Ecosul": 0,
    "accident_place_Ecovias_do_Araguaia": 1,
    "accident_place_Ecovias_do_Cerrado": 0,
    "accident_place_MSVIA": 0,
    "accident_place_Novadutra": 0,
    "accident_place_RIOSP": 0,
    "accident_place_Rodovia_do_Aço": 0,
    "accident_place_Transbrasiliana": 0,
    "accident_place_VIA040": 0,
    "accident_place_Via_Bahia": 1,
    "accident_place_Via_Brasil": 0,
    "accident_place_Via_Costeira": 0,
    "accident_place_Via_Sul": 0
}

````

**Saída**:
- Predição do modelo (ex.: `{"prediction": 0 ou 1, "result": "Acidente severo ou não severo""}`)

## Vídeo Demonstrativo

[Link para o vídeo demonstrativo do funcionamento do modelo em produção](https://youtu.be/_uU2IMef9C0) (https://youtu.be/_uU2IMef9C0)


## Como Executar a Aplicação no EC2 da AWS

### Pré-requisitos:
- Ter uma instância EC2 na AWS.
- Acesso SSH à instância EC2 com o arquivo `.pem` correspondente.
- Imagem Docker disponível no DockerHub: `lyorrei/ai_fastapi_aws:v1`.

### Passo a Passo:

1. **Acessar sua instância EC2**:
   - Utilize o arquivo `.pem` do seu EC2 para conectar-se via SSH:
     ```bash
     ssh -i "path/to/your-key.pem" ec2-user@YOUR-EC2-IP
     ```

2. **Instalar o Docker**:
   - Atualize os pacotes da instância:
     ```bash
     sudo yum update -y
     ```
   - Instale o Docker:
     ```bash
     sudo yum install docker -y
     ```
   - Inicie o serviço Docker:
     ```bash
     sudo service docker start
     ```
   - Adicione o usuário `ec2-user` ao grupo `docker`:
     ```bash
     sudo usermod -a -G docker ec2-user
     ```

3. **Desconecte e reconecte-se ao EC2**:
   Para que as mudanças de grupo sejam refletidas, saia da sessão SSH e reconecte-se.

4. **Puxe a imagem Docker do DockerHub**:
   - Execute o seguinte comando para obter a imagem:
   ```bash
   docker pull lyorrei/ai_fastapi_aws:v1
   ```

5. **Rodar o container**:
   - Inicie a aplicação com:
   ```bash
   docker run -d -p 8000:8000 lyorrei/ai_fastapi_aws:v1
   ```

6. **Configurar o Grupo de Segurança do EC2**:
   - Vá ao console da AWS.
   - No EC2 Dashboard, clique em “Security Groups”.
   - Selecione o Grupo de Segurança da sua instância.
   - Em “Inbound rules”, adicione uma regra que permita tráfego TCP na porta 8000 de qualquer IP (ou de IPs específicos, se preferir).

7. **Acessar a aplicação**:
   - Com um navegador ou cliente HTTP, vá para:
   ```
   http://YOUR-EC2-IP:8000

   # Exemplo
   http://3.87.88.55:8000/docs
   ```
