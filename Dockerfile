FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04 AS base-build
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        python3.12 python3.12-venv python3-pip \
        build-essential \
        git git-lfs curl \
        libsndfile1 \
        libsox-dev libsox-fmt-all sox \
        libgl1 libglib2.0-0 \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/*


FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04 AS base-runtime
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        python3.12 \
        git git-lfs \
        libsndfile1 \
        libsox3 sox \
        libgl1 libglib2.0-0 \
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
WORKDIR /home/user/app

RUN python3.12 -m venv /home/user/app/venv && \
    pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.9.1+cu128 \
    torchaudio==2.9.1+cu128 \
    torchvision==0.24.1+cu128

COPY ./wheel /home/user/app/wheel
COPY ./requirements_linux.txt /home/user/app/requirements_linux.txt

# Core app dependencies (DramaBox + MMAudio + utilities)
RUN pip install --no-cache-dir -r /home/user/app/requirements_linux.txt

# Runtime extras used by active features
RUN pip install --no-cache-dir \
    deepfilternet==0.5.6 \
    qwen-asr \
    openai-whisper \
    "resemble-perth @ git+https://github.com/resemble-ai/Perth.git@master" \
    "transformers>=4.45.0" \
    "peft>=0.7.0"


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

WORKDIR /home/user/app
COPY --chown=1001:1001 --from=builder /home/user/app/venv /home/user/app/venv

COPY ./modules /home/user/app/modules
COPY ./wheel /home/user/app/wheel
COPY ./docs /home/user/app/docs
COPY ./tests /home/user/app/tests
COPY ./voice_clone_studio.py /home/user/app/voice_clone_studio.py
COPY ./config.json /home/user/app/config.json
COPY ./emotions.json /home/user/app/emotions.json
COPY ./prompts.json /home/user/app/prompts.json

EXPOSE 7860
CMD ["python3", "voice_clone_studio.py"]
