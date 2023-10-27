#!/bin/bash

hosts=(
    slave0
    slave1
    slave2
    slave3
    slave4
    slave5
    slave6
    slave7
)

for host in ${hosts[@]}; 
do
    echo Syncing host: $host
    # scp -r /root/Anns_Blancer/Conf $host:/home
    scp -r /root/Anns_Blancer/MiniIVF/bin $host:/home/anns_balancer/cmake


    # ssh $host "mkdir -p /home/anns/dataset/sift1m /home/anns/index/sift1m /home/anns/query/sift1m"

    # ssh $host "rm -rf /home/anns/query/sift1m"
    # ssh $host "mkdir -p /home/anns/query/sift1m"
    # scp -r /RF/query/sift1m $host:/home/anns/query/

    # ssh $host "rm -rf /home/anns/index/sift1m"
    # ssh $host "mkdir -p /home/anns/index/sift1m"
    # scp -r /RF/index/sift1m/nt100k_pq128_kc2000/ $host:/home/anns/index/sift1m/
    
    # ssh $host "rm -rf /home/anns/dataset/sift1m"
    # ssh $host "mkdir -p /home/anns/dataset/sift1m"
    # scp -r /RF/dataset/sift1m/nt100k_pq128_kc2000 $host:/home/anns/dataset/sift1m
done