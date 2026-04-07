FROM python:3.10-slim

# Create a non-root user for Hugging Face Spaces (UID 1000 is required)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy the entire project
COPY --chown=user . /app

# Install the project and its dependencies (from root pyproject.toml)
RUN pip install --no-cache-dir .

# Hugging Face Spaces require port 7860
CMD ["uvicorn", "scheduler.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
