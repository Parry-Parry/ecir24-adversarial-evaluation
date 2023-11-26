#!/usr/bin/bash

docker run \
	--rm -ti \
	-w /app \
	-v ${PWD}:/app \
	--entrypoint ./token-search.sh \
	registry.webis.de/code-research/tira/tira-user-tira-ir-starter/pygaggle:0.0.1-monot5-base-msmarco-10k

