FROM python:3.10-slim

# Create a non-root user for Hugging Face Spaces
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV ENABLE_WEB_INTERFACE=true

WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy the application code
COPY --chown=user . /app

# Install the scheduler package and its dependencies
RUN pip install --no-cache-dir .

# Command required by Hugging Face Spaces: run uvicorn on port 7860
CMD ["uvicorn", "scheduler.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
