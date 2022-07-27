FROM python:3.10

WORKDIR /project

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . /project/

WORKDIR /project/app

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "80"]