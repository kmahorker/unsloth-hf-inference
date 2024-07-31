set -e
IMAGE_NAME=unsloth-hf-inference
REPO_NAME=unsloth-hf-inference
IMAGE_TAG=latest
DOCKER_USER=<YOUR_DOCKERHUB_USERNAME>
HOST=<YOUR_DOCKERHUB_USERNAME>

docker login -u $DOCKER_USER
docker build --platform linux/amd64 -t $IMAGE_NAME:$IMAGE_TAG .
docker tag $IMAGE_NAME:$IMAGE_TAG $HOST/$REPO_NAME:$IMAGE_TAG

docker push $HOST/$REPO_NAME:$IMAGE_TAG