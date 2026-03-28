FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV and video processing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git wget unzip build-essential cmake \
    libgl1-mesa-dev ffmpeg \
    libavcodec-dev libavformat-dev libswscale-dev \
    libgstreamer1.0-0 gstreamer1.0-plugins-base \
    libjpeg-dev libpng-dev libtiff-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV OPENCV_IO_ENABLE_NONFREE=0
ENV QT_QPA_PLATFORM=offscreen

COPY requirements.txt .

# Install pip tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch CPU (2.0.1 has ARM support)
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Install OpenCV headless
RUN pip install --no-cache-dir opencv-python-headless

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Clone the model repository
RUN git clone https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection.git model_code

# Download model weights from Google Drive with retries
RUN mkdir -p /app/model_code/gdrive_weights && \
    for i in 1 2 3; do \
        gdown --folder https://drive.google.com/drive/folders/19bNOs8_rZ7LmPP3boDS3XvZcR1iryHR1 -O /app/model_code/gdrive_weights --remaining-ok && break; \
        echo "gdown attempt $i failed, retrying in 10s..."; \
        sleep 10; \
    done

# Set up weight directories and move files
RUN mkdir -p /app/model_code/efficient-vit/pretrained_models && \
    mkdir -p /app/model_code/cross-efficient-vit/pretrained_models

RUN if [ -f /app/model_code/gdrive_weights/efficient_vit.pth ]; then \
        mv /app/model_code/gdrive_weights/efficient_vit.pth /app/model_code/efficient-vit/pretrained_models/efficient_vit.pth; \
    else \
        echo "Warning: efficient_vit.pth not found in gdrive_weights"; \
    fi && \
    if [ -f /app/model_code/gdrive_weights/cross_efficient_vit.pth ]; then \
        mv /app/model_code/gdrive_weights/cross_efficient_vit.pth /app/model_code/cross-efficient-vit/pretrained_models/cross_efficient_vit.pth; \
    else \
        echo "Warning: cross_efficient_vit.pth not found in gdrive_weights"; \
    fi && \
    rm -rf /app/model_code/gdrive_weights

# Verify critical weight file exists (warn only, don't fail build)
RUN if [ ! -f /app/model_code/cross-efficient-vit/pretrained_models/cross_efficient_vit.pth ]; then \
        echo "WARNING: cross_efficient_vit.pth not found! Model won't work until weights are provided." ; \
    fi

COPY app.py .

ENV PYTHONPATH=/app/model_code/cross-efficient-vit:/app/model_code/efficient-vit:/app/model_code/preprocessing:/app/model_code/cross-efficient-vit/efficient_net:${PYTHONPATH}

ENV MODEL_PORT=7001
ENV PRELOAD_MODEL="false"
ENV MODEL_TIMEOUT="900"
ENV USE_GPU="false"
ENV DEFAULT_MODEL_VARIANT="cross_efficient_vit"
ENV CROSS_EFFICIENT_VIT_MODEL_PATH="/app/model_code/cross-efficient-vit/pretrained_models/cross_efficient_vit.pth"
ENV EFFICIENT_VIT_MODEL_PATH="/app/model_code/efficient-vit/pretrained_models/efficient_vit.pth"
ENV CROSS_EFFICIENT_VIT_CONFIG_PATH="/app/model_code/cross-efficient-vit/configs/architecture.yaml"
ENV EFFICIENT_VIT_CONFIG_PATH="/app/model_code/efficient-vit/configs/architecture.yaml"

EXPOSE ${MODEL_PORT}
CMD ["python", "app.py"]
