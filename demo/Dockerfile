FROM python:3.12-slim

WORKDIR /app
COPY fake_agent/app.py /app/app.py
COPY smoke.py /app/smoke.py

CMD ["python", "/app/app.py", "--host", "0.0.0.0", "--port", "18091"]
