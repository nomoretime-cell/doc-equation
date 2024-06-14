FROM python:3.11.5

WORKDIR /doc-equation
# copy project
COPY model/ /doc-equation/model
COPY equation/ /doc-equation/equation
COPY *.py poetry.lock pyproject.toml entrypoint.sh ./
COPY prod.env local.env
COPY nltk.tar.gz /tmp/nltk.tar.gz
RUN mkdir -p /root/nltk_data && \
    tar xzf /tmp/nltk.tar.gz -C /

# install poetry
# RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir poetry nltk gunicorn uvicorn
# install poetry dependencies
RUN poetry config installer.max-workers 20
# RUN poetry install --no-root --no-dev
RUN poetry install --no-interaction --no-ansi -vvv --no-root --no-dev && \
    rm -rf /root/.cache/pypoetry/artifacts /root/.cache/pypoetry/cache
    
CMD ["./entrypoint.sh"]

