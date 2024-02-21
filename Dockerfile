FROM python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

COPY ./app/algorithm /code/app/algorithm

ENV HOST="0.0.0.0"

ENV PORT=5000

ENTRYPOINT uvicorn app.app:app --host ${HOST} --port ${PORT}
