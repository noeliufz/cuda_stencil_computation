#!/bin/bash

# 数据行数组
  data=(
      "32,32,1,1024"
  "32,32,16,64"
  "32,32,32,32"
  "32,32,64,16"
  "32,32,1024,1"
)


# 遍历数据行
for line in "${data[@]}"; do
  # 用逗号分割行
  IFS=',' read -r -a params <<< "$line"
  
  # 获取参数
  g1=${params[0]}
  g2=${params[1]}
  b1=${params[2]}
  b2=${params[3]}
  
  # 构建命令
  cmd="./testAdvect -g ${g1},${g2} -b ${b1},${b2} 2048 2048 100 "
  
  # 输出命令
  echo "Running: $cmd"
  
  # 执行命令
  $cmd

  # 输出一个空行
  echo
done

