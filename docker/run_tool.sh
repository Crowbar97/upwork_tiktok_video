# NOTE: this script is used
# just for running dockerized tool.

IMAGE_NAME="upwork_tiktok:tf_cpu"
CONTAINER_NAME="upwork_tiktok_tf_cpu"

pushd ..
docker run \
    -it --rm --privileged \
    -v ${PWD}:/project \
    --name $CONTAINER_NAME \
    $IMAGE_NAME \
    bash run.sh
popd