SHELL = /usr/bin/env bash
dockerfile = ./Dockerfile

all: ;

docker-build:
	docker build --no-cache -f $(dockerfile) . -t cryptocurrency-rnn:latest

docker-run:
	docker run -it --rm  -u $(id -u):$(id -g) \
	-v $(pwd):/app/notebooks \
	-p 8888:8888 cryptocurrency-rnn:latest

run:
	python src/main.py

clean:
	rm src/*.pyc

fclean: clean
	rm -f src/*.pyc

re: fclean all

.PHONY: clean fclean all re docker-build docker-run run
