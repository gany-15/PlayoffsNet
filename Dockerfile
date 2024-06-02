FROM --platform=amd64 python:3.10-slim
WORKDIR /playoffsnet
ENV HOST 0.0.0.0
RUN apt-get update
RUN apt-get install -y libhdf5-dev pkg-config gcc
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN pip install gunicorn ipython
COPY . .
EXPOSE 7000
CMD ["gunicorn", "main:nba", "-b", "0.0.0.0:7000", "-w", "4"]