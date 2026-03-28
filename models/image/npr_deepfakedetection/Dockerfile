FROM python:3.9-slim

WORKDIR /app

# Install git and wget
RUN apt-get update && apt-get install -y --no-install-recommends git wget && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install pip tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch CPU-only (works on both x86 and ARM)
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Clone the repository
RUN git clone https://github.com/chuangchuangtan/NPR-DeepfakeDetection.git npr_deepfakedetection

# Download model weights
RUN mkdir -p npr_deepfakedetection/weights && \
    wget -O npr_deepfakedetection/weights/NPR.pth \
    https://github.com/chuangchuangtan/NPR-DeepfakeDetection/raw/main/model_epoch_last_3090.pth

# Set environment variables
ENV MODEL_PORT=5001
ENV USE_GPU=false
ENV PRELOAD_MODEL=false
ENV MODEL_TIMEOUT=600

EXPOSE 5001

CMD ["python", "app.py"]
