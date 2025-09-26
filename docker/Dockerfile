FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*

# Install pipenv or other dependencies here if needed
# COPY Pipfile Pipfile.lock ./
# RUN pip install pipenv && pipenv install --system --deploy

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose the application port
EXPOSE 8000

# Default command (adjust if you use Gunicorn or other ASGI servers)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
