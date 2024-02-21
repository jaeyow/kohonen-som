- pyenv versions
- pyenv install 3.11.1
- pyenv local 3.11.1
- python -m venv kohonen-som-env
- source kohonen-som-env/bin/activate

pip install fastapi
pip install "uvicorn[standard]"

pip freeze > requirements.txt

docker build -t kohonen-som .
docker run -p 5000:5000 kohonen-som:latest
