#!/bin/bash

cid=`docker ps | grep huzaifa/workspace/yoga-lab:vivaldi | awk '{print $1}'`

docker exec -ti $cid /bin/bash