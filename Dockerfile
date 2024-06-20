FROM python:3.10-slim
WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install pandasai
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8097
CMD ["python", "llama_rag.py"]
