import os


def batch_convert_gbk_to_utf8(root_dir):
    """
    遍历指定目录及其所有子目录下的 CSV 文件，
    智能检测并把 GBK 编码的文件转换为 UTF-8 编码覆盖原文件。
    """
    print(f"🚀 开始扫描目录: {root_dir}\n")

    converted_count = 0
    skipped_count = 0
    error_count = 0

    # os.walk 会递归遍历 root_dir 下的所有子文件夹
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            # 仅处理 .csv 结尾的文件
            if filename.lower().endswith('.csv'):
                file_path = os.path.join(dirpath, filename)

                # 1. 智能检测：先尝试以 UTF-8 读取
                try:
                    with open(file_path, 'r', encoding='utf-8') as f_test:
                        f_test.read()
                    # 如果上面没报错，说明已经是 UTF-8，直接跳过
                    print(f"⏭️ 跳过 (已经是 UTF-8): {file_path}")
                    skipped_count += 1
                    continue
                except UnicodeDecodeError:
                    # 如果报错，说明是 GBK 或其他编码，进入下方转换流程
                    pass

                # 2. 转换流程：以 GBK 读取，以 UTF-8 覆写
                try:
                    # 读取
                    with open(file_path, 'r', encoding='gbk', errors='ignore') as f_in:
                        content = f_in.read()

                    # 覆写 (newline='' 防止 Windows 产生多余空行)
                    with open(file_path, 'w', encoding='utf-8', newline='') as f_out:
                        f_out.write(content)

                    print(f"✅ 成功转换: {file_path}")
                    converted_count += 1

                except Exception as e:
                    print(f"❌ 转换失败 [{file_path}]: {e}")
                    error_count += 1

    # 打印最终统计结果
    print("\n" + "=" * 40)
    print("🎉 批量转换任务完成！")
    print(f"📂 扫描总数: {converted_count + skipped_count + error_count} 个 CSV 文件")
    print(f"✅ 成功转换: {converted_count} 个")
    print(f"⏭️ 无需转换: {skipped_count} 个")
    print(f"❌ 发生错误: {error_count} 个")
    print("=" * 40)


if __name__ == "__main__":
    # 你的目标文件夹路径
    target_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed")

    # 建议在执行批量覆盖操作前，先在资源管理器里把 Data_CSV 文件夹复制备份一份！
    batch_convert_gbk_to_utf8(target_directory)