#!/bin/sh

echo $1 > VERSION && \
    sed -i "s/\"version\": \".*\"/\"version\": \"$1\"/" package.json && \
    sed -i "s/\"__version__\", \".*\"/\"__version__\", \"$1\"/" python/lib.c
