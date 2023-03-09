#!/bin/bash

docker build -t huzaifasuri/vivaldi:mayo_retina -f Dockerfile.hks .
docker push huzaifasuri/vivaldi:mayo_retina