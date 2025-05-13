#!/bin/bash

# 获取总内存大小（以 KB 为单位）
total_mem=$(free -k | awk '/Mem:/ {print $2}')

# 将 20G 转换为 KB（1G=1024*1024 KB）
target_mem=$((20 * 1024 * 1024))
target_script= "/home/yiruolei/project/AIGCdetector/SAFE/scripts/train.sh"
# 比较内存大小
if [ $total_mem -gt $target_mem ]; then
    echo "内存大于 ${target_mem}G，执行 ${target_script}"
    $target_script  # 替换为 a.sh 的实际路径
else
    echo "内存不大 于${target_mem}G"
fi