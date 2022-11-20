# FROM ubuntu:latest
FROM python:3.8.1
COPY mlops /exp/mlops
COPY ./requirements.txt /exp/requirements.txt
RUN pip install --no-cache-dir -r /exp/requirements.txt
WORKDIR /exp
CMD ["python", "./mlops/plot_digits_classification.py"]
