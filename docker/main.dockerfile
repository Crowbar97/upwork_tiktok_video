# This dockerfile is for AMD64 systems.

FROM tensorflow/tensorflow:2.13.0rc0

WORKDIR /project

RUN apt update && apt install -y python3-pip tmux tree \
    && pip install --upgrade pip

# This need for OpenCV
RUN apt install -y ffmpeg libsm6 libxext6 \
    && pip install opencv-python tqdm moviepy