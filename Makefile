.PHONY: build run

IMAGE := sql-agent:latest
CONTAINER := sql-agent

build:
	DOCKER_BUILDKIT=1 sudo docker build -t $(IMAGE) .

run:
	sudo docker run -d \
		--name $(CONTAINER) \
		-p 8000:8000 \
		-p 8501:8501 \
		--env-file .env \
		$(IMAGE)
