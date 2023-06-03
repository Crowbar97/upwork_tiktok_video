# This dockerfile is for Mac M1.

FROM armswdev/tensorflow-arm-neoverse:r23.05-tf-2.12.0-onednn-acl

WORKDIR /project

RUN sudo apt update && sudo apt install -y python3-pip tmux tree \
    && pip install --upgrade pip

# This need for OpenCV
RUN sudo apt install -y ffmpeg libsm6 libxext6 \
    && pip install opencv-python tqdm moviepy