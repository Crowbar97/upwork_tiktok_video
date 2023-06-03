IMAGE_NAME="upwork_tiktok:tf_cpu"

docker build -t $IMAGE_NAME \
             -f main.dockerfile \
             ../