import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

# 这里的路径改为你的实际路径
local_dir = "./models/bge-m3"

print("正在补全下载模型权重...")
snapshot_download(
    repo_id="BAAI/bge-m3",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    force_download=True,
    ignore_patterns=["*.DS_Store"] # 避开报错文件
)
print("✅ 下载完成，请重新运行你的检索程序。")