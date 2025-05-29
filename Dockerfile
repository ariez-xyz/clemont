FROM python:3.11-slim

# Install system dependencies for dd/CUDD
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install dd with CUDD
COPY install_dd.sh /tmp/
RUN chmod +x /tmp/install_dd.sh && /tmp/install_dd.sh

# Install your package
COPY . /app
WORKDIR /app
RUN pip install .

CMD ["python"]
