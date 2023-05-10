#!/bin/bash

# docker build -t huzaifasuri/vivaldi:mayo_retina -f Dockerfile.hks .
# docker push huzaifasuri/vivaldi:mayo_retina

docker build -t huzaifasuri/vivaldi:fpn_5 -f Dockerfile.hks .
docker push huzaifasuri/vivaldi:fpn_5