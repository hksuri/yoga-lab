#!/bin/bash

# docker run --rm --gpus=all --shm-size 60G --ipc=host -v /home/huzaifa:/home/huzaifa -v /home/john:/home/john -v /home/neeraj:/home/neeraj -v /mnt/ssd_2tb_1:/mnt/ssd_2tb_1 -v /mnt/ssd_4tb_0:/mnt/ssd_4tb_0 -v /mnt/hdd_16tb_0:/mnt/hdd_16tb_0 -v /mnt/scratch:/mnt/scratch -v /shared:/shared -v /scr:/scr -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -w /home/huzaifa --user="$(id -u):$(id -g)" --volume="$PWD:/app" -p 8895:8895 -it huzaifasuri/vivaldi:fpn_5

docker run --rm --gpus=all --shm-size 90G --ipc=host -v /home/huzaifa:/home/huzaifa -v /mnt/ssd_2tb_1:/mnt/ssd_2tb_1 -v /mnt/ssd_4tb_0:/mnt/ssd_4tb_0 -v /mnt/hdd_16tb_0:/mnt/hdd_16tb_0 -v /scr:/scr -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -w /home/huzaifa --user="$(id -u):$(id -g)" --volume="$PWD:/app" -p 8892:8892 -it huzaifasuri/vivaldi:mayo_retina
