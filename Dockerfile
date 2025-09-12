# syntax=docker/dockerfile:1
# FROM python:3.8-slim-buster
FROM python:3.14.0rc1-alpine3.22
LABEL maintainer="Thomas"

WORKDIR /python-docker

COPY ./src/*.py /python-docker/src
COPY ./pyproject.toml /python-docker/pyproject.toml
RUN uv sync

EXPOSE 80

CMD [ "uv", "run", "./src/interface.py"]

