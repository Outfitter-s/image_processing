FROM python:3.11.14-slim AS  builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt

FROM python:3.11.14-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:5000"]
