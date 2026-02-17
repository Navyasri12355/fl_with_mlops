FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
# Add missing dependencies to a temporary requirements file or update the existing one
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    prefect \
    mlflow \
    tensorflow \
    flwr \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    h5py \
    tqdm \
    python-multipart

# Copy the rest of the application
COPY . .

# Expose ports: FastAPI (8000), MLflow (5000), Prefect (4200)
EXPOSE 8000 5000 4200

# Entry point will be handled by docker-compose
CMD ["python", "web_backend.py"]
