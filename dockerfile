FROM python:3.10-slim

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y build-essential
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m botuser
USER botuser

CMD ["python", "src/telegram_agent.py"]
