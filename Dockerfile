# syntax=docker/dockerfile:1
FROM astral/uv:python3.12-bookworm-slim
LABEL maintainer="Thomas"

WORKDIR /python-docker

COPY ./src/main.py src/main.py
COPY ./src/interface.py src/interface.py
COPY ./src/aiexpertlawyer.py src/aiexpertlawyer.py
COPY ./src/mytools.py src/mytools.py
COPY ./pyproject.toml ./pyproject.toml
COPY ./interface/index.html interface/index.html
RUN uv sync

COPY ./chroma_langchain_db /chroma_langchain_db

EXPOSE 8000

CMD [ "uv", "run", "./src/interface.py"]

