#!/usr/bin/bash

docker run \
	--rm -ti \
	-v ${PWD}:/app \
	-w /app \
	--entrypoint /app/score-t5.py \
	registry.webis.de/code-research/tira/tira-user-tira-ir-starter/pygaggle:0.0.1-monot5-base-msmarco-10k \
	--input /app/t5-base-re-ranking --output /app/t5-base-re-ranking

