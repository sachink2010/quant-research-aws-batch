FROM pytorch/pytorch:latest

# Set working directory
WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    boto3 \
    mlflow

# Copy source code
COPY src/ /app/src/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set the entrypoint
ENTRYPOINT ["python", "src/main.py"]
