# Use 3.11-slim for the best balance of speed and compatibility
FROM python:3.11-slim

# Install system dependencies
# libgl1 and libglib2.0-0 are REQUIRED for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libpq-dev \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create user for Hugging Face
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Download model
RUN mkdir -p videos/models && \
    curl -L -o videos/models/pose_landmarker_heavy.task \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task

# Install Python requirements
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
# Use --prefer-binary to ensure we don't try to compile things from scratch
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy application code
COPY --chown=user . .

EXPOSE 7860

CMD ["python", "manage.py", "runserver", "0.0.0.0:7860"]