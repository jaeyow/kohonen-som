# Kohonen Self-Organizing Maps (SOM) with FastAPI

This is a Python implementation of [Kohonen Self-Organizing Maps (SOM)](https://en.wikipedia.org/wiki/Self-organizing_map), a type of unsupervised learning algorithm. Kohonen Maps are typically used for clustering and visualising so that higher-dimensional data can be represented in lower dimensions, typically in 2D like in a rectangular topology or grid.

In addition to segmentation and clustering analysis, it is also a form of dimensionality reduction technique so that the high-dimensional data in the input layer can be represented in the output grid. 

![Network](http://www.pitt.edu/~is2470pb/Spring05/FinalProjects/Group1a/tutorial/kohonen1.gif)

What's great about studying Kohonen Maps is that they are relatively simple algorithms, and can seem like magic when they are able to cluster and segment the input data. In this project, we are presenting the Kohonen SOM with input data in the form or RGB colours, and it will try to segment and cluster the colours in a 2D grid.

Because our input data is made up of 3 components (features), the nodes in the output layer are also made up of the same RGB components making them easy and fun to visualise. 

A description for training a Kohonen SOM algorithm is [detailed here](./kohonen.ipynb).

## Applications and Use Cases

A more popular clustering and segmentation algorithm is the [K-Means algorithm](https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/), which is also a type of unsupervised learning algorithm. So applications and use cases for K-Means can also be applied to Kohonen SOM, such as:

### Customer Segmentation

![Customer Segmentation](customer-segment-analysis-marketing-advertising-600nw-2134063767.webp)

Given the features of  customers, for example age, income, and race, we can segment them into different groups, and then target them with different marketing strategies.

### Image Compression

Although it maybe computationally expensive, a Kohonen SOM can be used to compress images, by reducing the dimensions of the image, while still retaining its important features.

### Recommender Systems

An online shopping platform can utilise users preferences to group them into different clusters and then recommend products based on the preferences of other users in the same cluster.

## Installation

It is best practice to use a virtual environment when working with Python. [pyenv](https://github.com/pyenv/pyenv) was used to manage the Python version, while [venv](https://docs.python.org/3/library/venv.html) was used to create the virtual environment. The following commands were used to create the virtual environment and install the required dependencies:

```bash
pyenv install 3.11.1
pyenv local 3.11.1
python -m venv kohonen-som-env
source kohonen-som-env/bin/activate
```

The following dependencies were installed using pip:
```bash
pip install fastapi
pip install "uvicorn[standard]"
pip install numpy
pip install matplotlib
pip install mangum
```
Then finally, the requirements.txt file was created using the following command:

```bash
pip freeze > requirements.txt
```

## Run locally on docker
- First we need to build the docker image
`docker build -t kohonen-som .`

- Then we can run the container like so:
`docker run -p 5000:5000 kohonen-som:latest`

- You can then try out the API by visiting the following URL:
[Kohonen-SOM Swagger on localhost](http://0.0.0.0:5000/docs)

- [FastAPI](https://fastapi.tiangolo.com/) will provide a Swagger/OpenAPI documentation for the API, which is very useful for testing and debugging, so you won't even need a client like Postman to test the API anymore.

## Swagger/OpenAPI documentation
- http://0.0.0.0:5000/docs

This is the image of the swagger documentation:
![Swagger](/images/swagger-openapi.png)

## Running on Jupyter
I have prepared a Jupyter notebook to demonstrate the Kohonen Self-Organizing Maps (SOM) algorithm. The notebook is available in the following link:
[Jupyter Notebook](kohonen.ipynb)

**Vectorised implementation using numpy**
In my first attempt at implementing the Kohonen SOM algorithm, I used the typical Python nested loops following the algorithm [described here](./kohonen.ipynb) to the letter. However, I quickly realised that increasing the iterations to 200, 500, 1000 or more would slow it to a crawl, not very exciting when deploying it to production.

The algorithm could be vectorised using numpy, which would make it more efficient, and faster. I have implemented both versions and compared the execution times, as shown below. The vectorised implementation is around **12x** faster than the non-vectorised version.

- Input layer: 20 colours
- Output layer: 100x100 grid
- Iterations: 1000
- Execution time (hh:mm:ss.ms): 00:00:09.294 (around 9 seconds)
![Jupyter](/images/vectorised-1000.png)

**Non-vectorised implementation using nested loops**
- Input layer: 20 colours
- Output layer: 100x100 grid
- Iterations: 1000
- Execution time (hh:mm:ss.ms): 00:01:48.90 (nearly two minutes!)

## Deploy to AWS

To facilitate the deployment to AWS Lambda, I have created a Dockerfile that uses the AWS Lambda Python 3.11 runtime. The required dependencies will surely be over the 250mb uncompressed limit even when using multiple [Lambda Layers](https://docs.aws.amazon.com/lambda/latest/dg/gettingstarted-concepts.html#gettingstarted-concepts-layer), so I've decided to use [Lambda Container Images](https://docs.aws.amazon.com/lambda/latest/dg/images-create.html), to allow us up to 10GB  container size limit for our API. [The Dockerfile is available here](./Dockerfile).

To Deploy our Lambda API to AWS, we have a few options, from Click-Ops, AWS [Serverless Application Model (SAM)](https://aws.amazon.com/serverless/sam/), or even Amplify, among others. For this type of project, I will probably use SAM, as it is simple and easy enough to integrate with [Github Actions](https://github.com/features/actions), should we want to add CI automation later on.

