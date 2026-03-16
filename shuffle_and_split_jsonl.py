import json
import os
import argparse
import random
import glob
import shutil
import array
import re
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict
from tqdm import tqdm

# =========================
# 1. 比例配置区域 (总和必须为 1.0)
# =========================
TARGET_CONFIG = {
    "early": {"min": 1, "max": 3, "ratio": 0.15, "name": "序盘(1-3巡)"},
    "mid_pre": {"min": 4, "max": 6, "ratio": 0.20, "name": "构筑(4-6巡)"},
    "mid_mid": {"min": 7, "max": 12, "ratio": 0.30, "name": "对攻(7-12巡)"},
    "late": {"min": 13, "max": 99, "ratio": 0.35, "name": "尾巡(13+巡)"}
}

# 预编译正则，匹配文本中的 "(第x巡"
TURN_PATTERN = re.compile(r"\(第(\d+)巡")


def get_bucket_key(turn: int) -> str:
    """根据巡目返回对应的桶键名"""
    if turn <= 3:
        return "early"
    elif turn <= 6:
        return "mid_pre"
    elif turn <= 12:
        return "mid_mid"
    else:
        return "late"


def worker_process_files(worker_id: int, file_paths: List[str], temp_dir: str) -> Dict[str, int]:
    """
    工作进程：
    1. 读取分配到的文件
    2. 解析 text 中的巡目
    3. 将行直接写入对应的临时分类文件 (流式写入，不占内存)
    4. 返回统计数据
    """
    # 为该进程创建专属的4个临时文件句柄，避免锁竞争
    handles = {}
    counts = {k: 0 for k in TARGET_CONFIG}

    try:
        # 打开临时文件 (二进制追加模式 wb)
        for key in TARGET_CONFIG:
            path = os.path.join(temp_dir, f"worker_{worker_id}_{key}.tmp")
            handles[key] = open(path, 'wb')  # 使用 buffer 默认即可

        for fp in file_paths:
            try:
                with open(fp, 'rb') as f_in:
                    for line_bytes in f_in:
                        line_strip = line_bytes.strip()
                        if not line_strip: continue

                        try:
                            # 仅解码用于正则匹配，提高速度
                            # 为了稳妥，这里对整行解码。如果追求极致性能，可只解码前 500 字节。
                            line_str = line_strip.decode('utf-8', errors='ignore')

                            # 正则提取巡目 (比 json.loads 快)
                            match = TURN_PATTERN.search(line_str)
                            if match:
                                turn = int(match.group(1))
                                key = get_bucket_key(turn)

                                # 写入原始 bytes，末尾加换行
                                handles[key].write(line_strip + b'\n')
                                counts[key] += 1
                        except Exception:
                            continue
            except Exception as e:
                print(f"[Worker {worker_id}] 读取出错: {fp} - {e}")

    finally:
        for f in handles.values():
            f.close()

    return counts


def merge_worker_files(temp_dir: str, keys: List[str], num_workers: int):
    """将分散的 worker 文件合并为最终的 4 个大临时文件"""
    print("\n[2/5] 正在合并临时分片...")
    for key in keys:
        final_path = os.path.join(temp_dir, f"{key}.jsonl")
        # 如果文件已存在先删除，防止追加
        if os.path.exists(final_path):
            os.remove(final_path)

        with open(final_path, 'wb') as f_out:
            for i in range(num_workers):
                worker_path = os.path.join(temp_dir, f"worker_{i}_{key}.tmp")
                if os.path.exists(worker_path):
                    with open(worker_path, 'rb') as f_in:
                        shutil.copyfileobj(f_in, f_out)
                    # 合并完立即删除，节省空间
                    os.remove(worker_path)


def build_compact_index(filepath: str) -> array.array:
    """建立内存高效的文件偏移索引 (只存位置，不存内容)"""
    # 'Q' 代表 unsigned long long (8 bytes)，1亿条数据仅占 760MB 内存
    offsets = array.array('Q')
    if not os.path.exists(filepath):
        return offsets

    with open(filepath, 'rb') as f:
        while True:
            offset = f.tell()
            if not f.readline():
                break
            offsets.append(offset)
    return offsets


def main():
    parser = argparse.ArgumentParser(description="立直麻将数据集重组工具")
    parser.add_argument("input_path", type=str, help="输入文件夹路径")
    parser.add_argument("--output_dir", type=str, default="dataset_balanced", help="输出文件夹")
    parser.add_argument("--lines_per_file", type=int, default=10000, help="每个输出文件包含的数据行数")
    parser.add_argument("--max_files", type=int, default=0, help="最大输出文件数 (0=自动根据短板计算)")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="并行处理进程数")
    parser.add_argument("--split_only", action="store_true",help="只按巡目分桶输出 early/mid_pre/mid_mid/late 四个jsonl，不做比例重组")

    args = parser.parse_args()

    # 校验比例和
    total_ratio = sum(c['ratio'] for c in TARGET_CONFIG.values())
    if abs(total_ratio - 1.0) > 0.001:
        print(f"错误：目标比例总和不为 100% (当前: {total_ratio:.2%})")
        return

    # 0. 环境准备
    temp_dir = "temp_rebalance_workspace"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    input_files = []
    if os.path.isfile(args.input_path):
        input_files = [args.input_path]
    elif os.path.isdir(args.input_path):
        input_files = glob.glob(os.path.join(args.input_path, "*.jsonl"))

    if not input_files:
        print("错误：未找到输入文件")
        return

    # 1. 第一阶段：并行分类 (Map & Scatter)
    chunk_size = math.ceil(len(input_files) / args.workers)
    file_chunks = [input_files[i:i + chunk_size] for i in range(0, len(input_files), chunk_size)]
    actual_workers = len(file_chunks)

    print(f"[1/5] 数据扫描与分类 ({actual_workers} 进程)...")
    total_counts = {k: 0 for k in TARGET_CONFIG}

    with ProcessPoolExecutor(max_workers=actual_workers) as executor:
        futures = []
        for i, chunk in enumerate(file_chunks):
            futures.append(executor.submit(worker_process_files, i, chunk, temp_dir))

        for future in tqdm(as_completed(futures), total=len(futures), unit="task"):
            res = future.result()
            for k, v in res.items():
                total_counts[k] += v

    print("\n--- 原始数据统计 ---")
    for k, v in total_counts.items():
        print(f"  {TARGET_CONFIG[k]['name']}: {v:,} 条")

    # 2. 第二阶段：合并 (Merge)
    merge_worker_files(temp_dir, list(TARGET_CONFIG.keys()), actual_workers)
    if args.split_only:
        print("\n[split_only] 输出四个分桶文件到 output_dir ...")
        for k in TARGET_CONFIG:
            src = os.path.join(temp_dir, f"{k}.jsonl")
            dst = os.path.join(args.output_dir, f"{k}.jsonl")
            shutil.copyfile(src, dst)
            print(f"  - {TARGET_CONFIG[k]['name']} -> {dst}")

        print("\n清理临时文件...")
        shutil.rmtree(temp_dir)
        print(f"完成！分桶文件已输出至: {args.output_dir}")
        return
    # 3. 第三阶段：建立索引与洗牌 (Index & Shuffle)
    print("\n[3/5] 建立索引与全局洗牌...")
    bucket_offsets = {}

    for k in TARGET_CONFIG:
        temp_file = os.path.join(temp_dir, f"{k}.jsonl")
        print(f"  正在索引 {TARGET_CONFIG[k]['name']} ... ", end="", flush=True)
        offsets = build_compact_index(temp_file)
        print(f"完成。索引数: {len(offsets):,}")

        # 转换为 list 进行 shuffle
        offset_list = list(offsets)
        random.shuffle(offset_list)
        bucket_offsets[k] = offset_list

    # 4. 第四阶段：计算配比与生成计划 (Calculate)
    print("\n[4/5] 计算生成计划...")

    # 4.1 计算单文件所需的各类型数量
    file_composition = {}
    for k, config in TARGET_CONFIG.items():
        needed = int(args.lines_per_file * config["ratio"])
        file_composition[k] = needed

    # 修正舍入误差，将剩余名额给占比最大的 late
    current_sum = sum(file_composition.values())
    if current_sum < args.lines_per_file:
        file_composition["late"] += (args.lines_per_file - current_sum)

    # 4.2 计算木桶效应 (由最缺数据的那个类别决定总产量)
    max_possible_files = float('inf')
    limit_bucket_name = ""

    print("  每文件所需构成:", file_composition)

    for k, needed in file_composition.items():
        if needed > 0:
            count = len(bucket_offsets[k])
            possible = count // needed
            print(f"    - {TARGET_CONFIG[k]['name']} 库存 {count} -> 可生成 {possible} 个文件")
            if possible < max_possible_files:
                max_possible_files = possible
                limit_bucket_name = TARGET_CONFIG[k]['name']

    final_count = max_possible_files
    if args.max_files > 0:
        final_count = min(final_count, args.max_files)

    print(f"\n  >>> 受限于 [{limit_bucket_name}]，最大可生成: {max_possible_files} 个文件")
    print(f"  >>> 实际计划生成: {final_count} 个文件")

    if final_count <= 0:
        print("错误：数据严重不足，无法生成任何完整文件。请检查输入数据量。")
        shutil.rmtree(temp_dir)
        return

    # 5. 第五阶段：生成最终文件 (Generate)
    print("\n[5/5] 开始生成数据集...")

    # 打开所有临时文件句柄
    read_handles = {}
    for k in TARGET_CONFIG:
        read_handles[k] = open(os.path.join(temp_dir, f"{k}.jsonl"), 'rb')

    pointers = {k: 0 for k in TARGET_CONFIG}

    try:
        for i in tqdm(range(final_count), unit="file"):
            output_content = []

            # 按比例抓取
            for k, count in file_composition.items():
                start = pointers[k]
                end = start + count

                # 获取偏移量切片
                batch_offsets = bucket_offsets[k][start:end]
                pointers[k] = end

                # 随机读取
                handle = read_handles[k]
                for offset in batch_offsets:
                    handle.seek(offset)
                    line = handle.readline()
                    output_content.append(line)

            # 文件内部再次洗牌，打乱各巡目的顺序
            random.shuffle(output_content)

            # 写入最终文件
            out_name = os.path.join(args.output_dir, f"train_balanced_{i:05d}.jsonl")
            with open(out_name, 'wb') as f_out:
                for line in output_content:
                    f_out.write(line)

    finally:
        for f in read_handles.values():
            f.close()
        print("\n清理临时文件...")
        shutil.rmtree(temp_dir)

    print(f"完成！数据已输出至: {args.output_dir}")


if __name__ == "__main__":
    main()