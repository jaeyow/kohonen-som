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

## Run locally on docker
- First we need to build the docker image
`docker build -t kohonen-som .`

- Then we can run it like so:
`docker run -p 5000:5000 kohonen-som:latest`

- You can then try out the API by visiting the following URL:
[Kohonen-SOM Swagger on localhost](http://0.0.0.0:5000/docs)

- FastAPI will provide a Swagger/OpenAPI documentation for the API, which is very useful for testing and debugging, so you won't even need a client like Postman to test the API anymore.

## Deploy to AWS

To facilitate the deployment to AWS Lambda, I have created a Dockerfile that uses the AWS Lambda Python 3.11 runtime. The required dependencies will surely be over the 250mb uncompressed limit even when using multiple [Lambda Layers](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-concepts.html#gettingstarted-concepts-layer), so I've decided to use [Lambda Container Images](https://docs.aws.amazon.com/lambda/latest/dg/images-create.html), to allow us up to 10GB  container size limit for our API. [The Dockerfile is available here](./Dockerfile).

To Deploy our Lambda API to AWS, we have a few options, from Click-Ops, SAM, or even Amplify. For this type of project, I will probably use SAM, as it is simple and easy enough to integrate with [Github Actions](https://github.com/features/actions), should we want to add CI automation later on.

## Swagger/OpenAPI documentation
- http://0.0.0.0:5000/docs

This is the image of the swagger documentation:
![Swagger](/images/swagger-openapi.png)