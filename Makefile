SHELL := /bin/bash
CURRENT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

install-latest-node:
	apt update \
	&& apt install -y nodejs npm \
	&& npm install n -g \
	&& n stable \
	&& apt purge -y nodejs npm \
	&& apt autoremove -y \
	&& ln -sf /usr/local/bin/node /usr/bin/node \

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

build:
	cd ${CURRENT_DIR}/view/txt2img \
	&& npm install \
	&& npm run build \
	&& cd ${CURRENT_DIR}/view/img2img \
	&& npm install \
	&& npm run build

launch:
	cd ${CURRENT_DIR} && python main.py
