#!/bin/bash

cid=`docker ps | grep huzaifasuri/vivaldi:mayo_retina | awk '{print $1}'`

docker exec -ti $cid /bin/bash