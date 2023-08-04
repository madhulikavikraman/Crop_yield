FROM python:3.9
WORKDIR /app
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ .
COPY data/cropsb.csv /app/data/cropsb.csv
VOLUME ["/app/data", "/app/model"]
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
