#!/bin/bash

# 找到所有的tar文件并解压，同时删除解压后的parquet和json文件，然后删除tar文件
for tar_file in *.tar; do
    # 解压tar文件
    tar -xvf "$tar_file" &&

    # 删除解压后的parquet文件
    find . -type f -name "*.parquet" -delete &&

    # 删除解压后的json文件
    find . -type f -name "*.json" -delete &&

    find . -type f -name "*.txt" -delete &&

    # 删除已解压的tar文件
    rm -f "$tar_file"
done
