#!/bin/bash

set -e

if [ "$#" -ne 2 ]; then
    echo "USAGE: dockerfiles/$0 [model_name] [tag number]"
    echo "e.g. dockerfiles/$0 DGII 1.1.2"
    exit 1
fi

WORK_PATH=`dirname $0`
MODEL=$1
TAG=$2

model_name_lc=`echo ${MODEL} | tr '[:upper:]' '[:lower:]'`
docker build -t ${model_name_lc}:${TAG} -f dockerfiles/Dockerfile.${MODEL} .
