#!/bin/sh

set -e

if [ "$#" -ne 2 ]; then
    echo "USAGE: $0 [model_name] [tag number]"
    exit 1
fi

WORK_PATH=`dirname $0`
MODEL=$1
TAG=$2

model_name_lc=`echo ${file_path##*.} | tr '[:upper:]' '[:lower:]'`
docker build -t ${model_name_lc}:${TAG} -f dockerfiles/Dockerfile.${MODEL} $WORK_PATH/../
