FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04 AS base-build
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        python3.12 python3.12-venv python3-pip \
        build-essential \
        git wget curl \
        libssl-dev libffi-dev \
        libsndfile1 \
        libsox-dev libsox-fmt-all sox \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/*
    

FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04 AS base-runtime
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        python3.12 \
        libssl3 \
        libffi8 \
        libsndfile1 \
        libsox3 \
        sox \
        git \
        git-lfs \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/*


FROM base-build AS user
RUN useradd -m -u 1001 -s /bin/bash user
USER user
ENV PATH="/home/user/app/venv/bin:$PATH" \
    HF_HOME=/home/user/app/.cache/huggingface \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1


FROM user AS builder
RUN python3.12 -m venv /home/user/app/venv && \
    pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.9.1+cu128 \
    torchaudio==2.9.1+cu128 \
    torchvision==0.24.1+cu128
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/home/user/.cargo/bin:${PATH}"
RUN pip install --no-cache-dir \
    gradio==6.5.0 \
    soundfile==0.13.1 \
    librosa==0.11.0 \
    numpy==1.26.4 \
    sox==1.5.0 \
    diffusers==0.36.0 \
    deepfilternet==0.5.6 \
    qwen-tts==0.0.5 \
    scipy \
    huggingface_hub \
    modelscope \
    psutil \
    packaging \
    wheel \
    ninja \
    openai-whisper \
    pytz \
    "flash_attn @ https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl" \
    bitsandbytes==0.49.1 \
    onnxruntime \
    onnxruntime-gpu==1.23.2 \
    markdown==3.10.1 \
    einops \
    https://github.com/csukuangfj/piper-phonemize/releases/download/2025.06.23/piper_phonemize-1.3.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl \
    git+https://github.com/ysharma3501/LinaCodec.git

COPY ./wheel /home/user/app/wheel
COPY ./requirements.txt /home/user/app/requirements.txt
WORKDIR /home/user/app
RUN pip install --no-cache-dir -r /home/user/app/requirements.txt
RUN rustup self uninstall -y


FROM base-runtime AS runtime
RUN useradd -m -u 1001 -s /bin/bash user
USER user
ENV PATH="/home/user/app/venv/bin:$PATH" \
    HF_HOME=/home/user/app/.cache/huggingface \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    BNB_CUDA_VERSION=128 \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

COPY --chown=1001:1001 --from=builder /home/user/app/venv /home/user/app/venv
COPY ./modules /home/user/app/modules
COPY ./wheel /home/user/app/wheel
COPY ./tests /home/user/app/tests
COPY ./docs /home/user/app/docs
COPY ./config.json /home/user/app/config.json
COPY ./voice_clone_studio.py /home/user/app/voice_clone_studio.py
WORKDIR /home/user/app
EXPOSE 7860
CMD ["python3", "voice_clone_studio.py"]
