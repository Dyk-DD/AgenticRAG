"""下载检索所需的嵌入模型到 <项目根>/models/。

用法：
    python scripts/download_model.py                  # 默认模型 bge-base-zh-v1.5（约 390 MB）
    python scripts/download_model.py --model bge-m3   # 换成 bge-m3（约 4.3 GB）
    python scripts/download_model.py --all            # 两个都下
    python scripts/download_model.py --force          # 忽略缓存强制重下（修复残缺下载）

默认走 hf-mirror.com 国内镜像；要用官方源就 `HF_ENDPOINT=https://huggingface.co`。
"""
import argparse
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import snapshot_download

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 键是 app/config.py 的 embedding_model 指向的目录名，值是 HuggingFace 上的仓库 id。
# bge-base-zh-v1.5 是 config.py 的默认值（768 维）；bge-m3 是可选替代，换用它
# 必须同时把 config.py 的 embedding_dimension 改成 1024，并重建向量索引 ——
# 两个模型的向量维度不同，混用会在插入时报维度不匹配。
MODELS = {
    "bge-base-zh-v1.5": "BAAI/bge-base-zh-v1.5",
    "bge-m3": "BAAI/bge-m3",
}
DEFAULT_MODEL = "bge-base-zh-v1.5"


def download(name: str, force: bool = False) -> None:
    local_dir = os.path.join(PROJECT_ROOT, "models", name)
    print(f"正在下载 {MODELS[name]} -> {local_dir}")
    snapshot_download(
        repo_id=MODELS[name],
        local_dir=local_dir,
        # 下成真实文件而不是指向 ~/.cache 的符号链接：models/ 会被挂进容器，
        # 指向宿主机缓存的软链在容器里是断的。
        local_dir_use_symlinks=False,
        force_download=force,
        ignore_patterns=["*.DS_Store"],  # 避开报错文件
    )
    print(f"✅ {name} 下载完成")


def main() -> None:
    parser = argparse.ArgumentParser(description="下载嵌入模型到 models/")
    parser.add_argument(
        "--model",
        choices=sorted(MODELS),
        default=DEFAULT_MODEL,
        help=f"要下载的模型（默认 {DEFAULT_MODEL}）",
    )
    parser.add_argument("--all", action="store_true", help="下载全部模型")
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略本地缓存强制重新下载（用于修复下载残缺）",
    )
    args = parser.parse_args()

    for name in sorted(MODELS) if args.all else [args.model]:
        download(name, force=args.force)


if __name__ == "__main__":
    main()
