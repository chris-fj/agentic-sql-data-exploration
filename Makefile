.PHONY: build run stop

IMAGE := sql-agent:latest
CONTAINER := sql-agent

stop:
	sudo docker rm -f $(CONTAINER) > /dev/null 2>&1 || 1
run: stop
	sudo docker compose up -d
	sudo docker compose logs -f
build-run: stop
	sudo docker compose up -d --build
	sudo docker compose logs -f
