IMAGE_NAME=deepspeech

# use branch name for tag if possible
IMAGE_TAG=$(shell git symbolic-ref --short -q HEAD || echo 'dev')

build: deps/ctcdecode deps/kenlm deps/warp-ctc
		sudo docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

deps/ctcdecode:
		git clone --recursive git@github.com:parlance/ctcdecode.git deps/ctcdecode

deps/kenlm:
		git clone git@github.com:kpu/kenlm.git deps/kenlm

deps/warp-ctc:
		git clone git@github.com:t-vi/warp-ctc.git deps/warp-ctc

clean:
		sudo docker images -q $(IMAGE_NAME):$(IMAGE_TAG) | \
				xargs --no-run-if-empty sudo docker rmi
		rm -rf deps

.PHONY: build clean
