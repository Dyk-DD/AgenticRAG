"""
下载 Monor/TCMNER 模型到本地 models 目录

用法：
    python scripts/download_tcmner.py
    python scripts/download_tcmner.py --model Monor/TCMNER  # 可选其他模型
    python scripts/download_tcmner.py --help

输出目录：<项目根>/models/TCMNER/
"""
import os
import sys
import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_DIR = os.path.join(PROJECT_ROOT, "models", "TCMNER")


def download_model(model_id: str, local_dir: str, force: bool = False):
    """下载 HuggingFace 模型到本地目录"""

    if os.path.exists(local_dir) and os.listdir(local_dir) and not force:
        existing = [f for f in os.listdir(local_dir) if f.endswith((".bin", ".safetensors", ".json"))]
        if existing:
            logger.info(f"模型目录已存在且非空 ({len(existing)} 个模型文件)，跳过下载")
            logger.info(f"  路径: {local_dir}")
            logger.info(f"  如需重新下载，请加 --force 参数")
            return True

    # 优先用 huggingface_hub.snapshot_download（更稳定的大文件下载）
    try:
        from huggingface_hub import snapshot_download

        logger.info(f"开始下载模型: {model_id}")
        logger.info(f"  目标路径: {local_dir}")

        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            ignore_patterns=["*.h5", "*.ot", "*.msgpack"],  # 排除非必要格式
        )

        logger.info(f"✅ 模型下载完成: {local_dir}")
        return True

    except ImportError:
        logger.warning("huggingface_hub 未安装，尝试用 transformers 下载...")

    # fallback: 用 transformers 下载（会先下载到 cache 再复制）
    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        logger.info(f"开始下载模型: {model_id}")

        # 下载并保存 tokenizer
        logger.info("下载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(local_dir)

        # 下载并保存模型（默认下载到 cache + 复制到 local_dir）
        logger.info("下载模型权重（这可能需要几分钟，取决于网络）...")
        model = AutoModelForTokenClassification.from_pretrained(model_id)
        model.save_pretrained(local_dir)

        # 保存配置文件
        model.config.save_pretrained(local_dir)

        logger.info(f"✅ 模型下载完成: {local_dir}")
        # 显示占用空间
        total_size = sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _, fn in os.walk(local_dir)
            for f in fn
        )
        logger.info(f"  模型大小: {total_size / 1024 / 1024:.1f} MB")
        return True

    except Exception as e:
        logger.error(f"❌ 下载失败: {e}")
        logger.error("请检查网络连接，或尝试设置 HF_ENDPOINT=https://hf-mirror.com")
        return False


def verify_model(local_dir: str) -> bool:
    """验证模型文件完整性"""
    required_files = [
        "config.json",
        "tokenizer.json",
    ]
    weight_files = [f for f in os.listdir(local_dir) if f.endswith((".bin", ".safetensors"))]

    found_files = os.listdir(local_dir)
    logger.info(f"\n模型目录内容 ({len(found_files)} 个文件):")
    for f in sorted(found_files):
        fpath = os.path.join(local_dir, f)
        size = os.path.getsize(fpath)
        logger.info(f"  {f:40s} {size / 1024 / 1024:.1f} MB")

    missing = [f for f in required_files if f not in found_files]
    if missing:
        logger.warning(f"缺少文件: {missing}")
        return False
    if not weight_files:
        logger.warning("未找到模型权重文件 (.bin 或 .safetensors)")
        return False
    return True


def test_inference(local_dir: str):
    """快速测试模型推理（可选 GPU）"""
    try:
        import torch
        from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

        device = 0 if torch.cuda.is_available() else -1
        if device == 0:
            logger.info("检测到 GPU，使用 CUDA 推理")
        else:
            logger.info("使用 CPU 推理")

        ner = pipeline(
            "ner",
            model=local_dir,
            tokenizer=local_dir,
            device=device,
        )

        test_cases = [
            "伯伯得了脂肪肝，平时比较胖",
            "高血压患者能吃党参吗",
            "我老婆体检发现乙肝大三阳",
        ]

        logger.info("\n--- 快速测试 ---")
        for text in test_cases:
            result = ner(text)
            entities = [(r["word"], r["entity"], f"{r['score']:.3f}") for r in result]
            logger.info(f"  输入: {text}")
            logger.info(f"  结果: {entities}")

        logger.info("✅ 模型推理测试通过")
        return True

    except Exception as e:
        logger.warning(f"测试推理失败（仅下载仍可正常使用）: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="下载 HuggingFace NER 模型到本地")
    parser.add_argument(
        "--model",
        default="Monor/TCMNER",
        help="HuggingFace 模型 ID (默认: Monor/TCMNER)",
    )
    parser.add_argument(
        "--out",
        default=TARGET_DIR,
        help=f"输出目录 (默认: {TARGET_DIR})",
    )
    parser.add_argument("--force", action="store_true", help="强制重新下载")
    parser.add_argument("--no-verify", action="store_true", help="跳过文件校验")
    parser.add_argument("--test", action="store_true", help="下载后运行推理测试")
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.out, exist_ok=True)

    # 镜像源提示
    mirror = os.environ.get("HF_ENDPOINT", "")
    if not mirror:
        logger.info(
            "提示：如果下载慢，可设置镜像源：\n"
            "  export HF_ENDPOINT=https://hf-mirror.com  (Linux/Mac)\n"
            "  set HF_ENDPOINT=https://hf-mirror.com     (Windows CMD)\n"
            "  $env:HF_ENDPOINT = 'https://hf-mirror.com'  (PowerShell)"
        )

    # 下载
    ok = download_model(args.model, args.out, force=args.force)
    if not ok:
        sys.exit(1)

    # 验证
    if not args.no_verify:
        ok = verify_model(args.out)
        if not ok:
            logger.warning("文件验证不完整，但下载已基本完成")

    # 测试
    if args.test:
        test_inference(args.out)

    print(f"\n✅ TCMNER 模型已保存到: {args.out}")


if __name__ == "__main__":
    main()
