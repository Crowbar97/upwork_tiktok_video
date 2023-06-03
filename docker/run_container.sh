# NOTE: this script is used for running
# container with development purposes.

IMAGE_NAME="upwork_tiktok:tf_cpu"
CONTAINER_NAME="upwork_tiktok_tf_cpu"

pushd ..
# -u $(id -u):$(id -g) \
# --runtime nvidia --gpus all \
docker run \
    -it -d --rm --privileged \
    -v ${PWD}:/project \
    --name $CONTAINER_NAME \
    $IMAGE_NAME \
    bash
popd