FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 7860

ENV API_BASE_URL=http://localhost:7860
ENV MODEL_NAME=farm-agent-v1

CMD ["python", "app.py"]