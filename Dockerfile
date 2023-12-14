FROM python:3.11
LABEL author="Vladimir Katin"

ENV HOST="0.0.0.0"
ENV PORT="8888"
ENV POETRY_VERSION="1.7.0"


EXPOSE $PORT

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir poetry==$POETRY_VERSION && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-root --without dev

ENV PYTHONPATH=/app

CMD poetry run uvicorn main:app --host $HOST --port $PORT
