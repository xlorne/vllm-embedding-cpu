#!/bin/bash
# download-model.sh

MODEL_NAME="BAAI/bge-m3"
LOCAL_DIR="./models/bge-m3"

echo "正在下载模型: $MODEL_NAME"
echo "目标目录: $LOCAL_DIR"

# 创建目录
mkdir -p "$LOCAL_DIR"

# 使用huggingface-cli下载（需先安装：pip install huggingface-hub）
python3 -c "
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='$MODEL_NAME',
    local_dir='$LOCAL_DIR',
    local_dir_use_symlinks=False,
    max_workers=4
)
"

echo "下载完成！目录结构："
find "$LOCAL_DIR" -type f | head -20
