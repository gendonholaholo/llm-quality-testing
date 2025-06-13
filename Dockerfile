FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if needed (uncomment if required)
# RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy metadata and install dependencies first for better caching
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && pip install .

# Copy the rest of the project
COPY llm_eval ./llm_eval
COPY scripts ./scripts
COPY configs ./configs
COPY tests ./tests
COPY data ./data
COPY results ./results

ENTRYPOINT ["llm-tester"] 