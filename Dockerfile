# --- Stage 1: Install Poetry & Configue -----
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as venv

# Updates & Packages
RUN apt-get update \
    && apt-get install -y \
         curl \
         python3.10 \
         python3.10-venv \
         python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Download and install poetry. Update Path
ENV POETRY_VERSION=1.6.1
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH "/root/.local/bin:$PATH"

# Copy pyproject.toml and poetry.lock for installing dependencies
WORKDIR /app
COPY pyproject.toml poetry.lock ./ 
RUN python3 -m venv --copies /app/venv

RUN . /app/venv/bin/activate && pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install poetry dependencies into the virtual environment
RUN . /app/venv/bin/activate && poetry install --only main


# --- Stage 2 -----
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as prod

# Updates & Packages
RUN apt-get update \
    && apt-get install -y \
         curl \
         python3.10 \
         python3.10-venv \
         python3-pip \
    && rm -rf /var/lib/apt/lists/*


# Set working dir
WORKDIR /app

# Copy the virtual environment and model from venv to prod
COPY --from=venv /app /app

ENV PATH /app/venv/bin:$PATH

RUN . /app/venv/bin/activate && pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Add code
COPY ./src /app/src

# Exposure port
EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
