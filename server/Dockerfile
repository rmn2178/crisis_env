FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app
ENV PYTHONPATH=/app

COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
