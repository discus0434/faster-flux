SHELL := /bin/bash
CURRENT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

build:
	cd ${CURRENT_DIR}/view/txt2img \
	&& npm install \
	&& npm run build \
	&& cd ${CURRENT_DIR}/view/img2img \
	&& npm install \
	&& npm run build
	&& cd ${CURRENT_DIR} \
	&& pip install -e .

launch:
	cd ${CURRENT_DIR} && python main.py

install-latest-node:
	apt update \
	&& apt install -y nodejs npm \
	&& npm install n -g \
	&& n stable \
	&& apt purge -y nodejs npm \
	&& apt autoremove -y \
	&& ln -sf /usr/local/bin/node /usr/bin/node \

sudo-install-latest-node:
	sudo apt update \
	&& sudo apt install -y nodejs npm \
	&& sudo npm install n -g \
	&& sudo n stable \
	&& sudo apt purge -y nodejs npm \
	&& sudo apt autoremove -y \
	&& sudo ln -sf /usr/local/bin/node /usr/bin/node

install-dependency:
	pip install --upgrade pip \
	&& pip install \
		torch \
		torchvision \
		--index-url https://download.pytorch.org/whl/nightly/cu121 \
	&& pip install \
		triton \
		torchao \
		--index-url https://download.pytorch.org/whl/cu121 \
	&& pip install git+https://github.com/huggingface/diffusers.git \
	&& pip install transformers accelerate peft nvitop sentencepiece \
		protobuf imagehash pydantic fastapi uvicorn

uninstall-old-torch:
	pip uninstall -y nvidia-cublas-cu12
	pip uninstall -y nvidia-cuda-cupti-cu12
	pip uninstall -y nvidia-cuda-nvrtc-cu12
	pip uninstall -y nvidia-cuda-runtime-cu12
	pip uninstall -y nvidia-cudnn-cu12
	pip uninstall -y nvidia-cufft-cu12
	pip uninstall -y nvidia-curand-cu12
	pip uninstall -y nvidia-cusolver-cu12
	pip uninstall -y nvidia-cusparse-cu12
	pip uninstall -y nvidia-nccl-cu12
	pip uninstall -y nvidia-nvjitlink-cu12
	pip uninstall -y nvidia-nvtx-cu12
	pip uninstall -y torch torchvision torchaudio
