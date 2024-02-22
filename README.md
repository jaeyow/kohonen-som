# Kohonen Self-Organizing Maps (SOM) with FastAPI
- blah

## Description
- blah

## Applications and Use Cases
-blah

## Installation
- pyenv versions
- pyenv install 3.11.1
- pyenv local 3.11.1
- python -m venv kohonen-som-env
- source kohonen-som-env/bin/activate

pip install fastapi
pip install "uvicorn[standard]"

pip freeze > requirements.txt

## Running on Jupyter
I have prepared a Jupyter notebook to demonstrate the Kohonen Self-Organizing Maps (SOM) algorithm. The notebook is available in the following link:
[Jupyter Notebook](kohonen.ipynb)

**Vectorised implementation using numpy**
- Input layer: 20 colours
- Output layer: 100x100 grid
- Iterations: 1000
- Execution time (hh:mm:ss.ms): 00:00:09.294 (around 9 seconds)
![Jupyter](/images/vectorised-1000.png)

**Non-vectorised implementation using nested loops**
- Input layer: 20 colours
- Output layer: 100x100 grid
- Iterations: 500
- Execution time (hh:mm:ss.ms): 00:01:00.90 (around 1 minute)
![Jupyter](/images/non-vectorised-500.png)


## Github Actions

- blah

## Swagger/OpenAPI documentation
- http://0.0.0.0:5000/docs

This is the image of the swagger documentation:
![Swagger](/images/swagger-openapi.png)

## Run locally on docker

docker build -t kohonen-som .
docker run -p 5000:5000 kohonen-som:latest

## Deploy to AWS
- blah