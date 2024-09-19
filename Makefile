SHELL := /bin/bash

install-latest-node:
	sudo npm install n -g \
		&& n stable \
		&& sudo apt purge -y nodejs npm \
		&& sudo apt autoremove -y \
		&& sudo ln -sf /usr/local/bin/node /usr/bin/node \

install-venv:
	pyenv local 3.11.9 \
	&& python -m venv .venv \
	&& source .venv/bin/activate \
	&& pip install --upgrade pip \
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
