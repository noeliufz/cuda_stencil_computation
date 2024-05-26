#!/bin/bash

# 数据数组
data=(
  "1,64"
  "2,32"
  "4,16"
  "8,8"
  "16,4"
  "32,2"
  "64,1"
)

# 遍历数据数组进行两两组合
for g in "${data[@]}"; do
  for b in "${data[@]}"; do
    # 构建命令
    cmd="./testAdvect -g ${g} -b ${b} 10000 10000 100 -o"
    
    # 输出命令
    echo "Running: $cmd"
    
    # 执行命令
    $cmd
    
    # 输出一个空行
    echo
  done
done
