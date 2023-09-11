# Usando uma imagem base oficial de Python
FROM python:3.10.13

# Definindo a pasta de trabalho no container
WORKDIR /app

# Copiando os arquivos necessários para a pasta de trabalho
COPY . .

# Instalando as bibliotecas necessárias
RUN pip install fastapi pandas pycaret uvicorn

# Comando para executar a aplicação
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
