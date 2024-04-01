# Experimental Models REPO
This repository is exclusively used to train new models that will be deployed into the backend chat processing subsystem. Once the models are trained and reaches quality standards to be deployed, the model .pth files could be copied into the ["Backend Chat Proccessing Subsystem" Repo](https://github.com/Alethianomous-AI/backend-chat-processing-subsystem).

# Dependencies Installation

## Installing via Docker
1. If running with Docker, please build the image using the `docker build -t <image_name> .`, where [image_name] is the name of the image.
2. Create a Docker container and run it by typing `docker run -it -e "$(pwd)":/code -p 8888:8888 <image_name> bash`. If you want to use the GPU, install [CUDA](https://developer.nvidia.com/cuda-downloads) and the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Then, type `docker run -it -e "$(pwd)":/code -p 8888:8888 --gpus all <image_name> bash` instead.
3. If you exit the Docker container, you can re-enter it by typing `docker container ps -a`, then typing `docker exec -it <container-id> bash` where [container_id] is the name of the container you have created.

## Using Python's Virtualenv
1. With the root directory of this repo as the current directory, type `python3 -m venv .env`.
2. On Linux, type `source activate .env/bin/activate`. If using Windows, type `.env\scripts\activate` in Command Prompt or `.env\scripts\activate.ps1` in PowerShell.
3. Install the dependencies by typing `python3 -m pip install -r requirements.txt`.
