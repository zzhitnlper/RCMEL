import json
import sys
import os

def calculate_metrics(json_file_path):

    # 初始化计数器
    total_entries = 0
    top1_correct = 0
    top2_correct = 0
    top3_correct = 0
    rank_sum = 0  # 用于计算 MR
    reciprocal_rank_sum = 0  # 用于计算 MRR

    # 读取 JSON 文件
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到文件 {json_file_path}")
        return None
    except json.JSONDecodeError:
        print(f"错误：文件 {json_file_path} 不是有效的 JSON 格式")
        return None

    # 遍历每条记录
    for entry in data:
        true_value = entry.get("true")
        # 舍弃 true = 0 的记录
        if true_value == 0:
            continue

        total_entries += 1  # 计数有效记录

        # 获取 top1_30 字段并拆分为列表
        top_list = entry["top1_30"].split()  # 分割成列表，例如 "1 2 3" -> ["1", "2", "3"]
        if true_value > len(top_list):  
            rank = len(top_list) + 1  # 如果超出范围，假设排名为最后一个位置之后
        else:
            rank = true_value  # 正确实体的排名（1-based index）

        # 计算 MR 和 MRR
        rank_sum += rank  # 累加排名用于 MR
        reciprocal_rank_sum += 1 / rank  # 累加倒数排名用于 MRR

        # 判断正确实体是否在 Top-k 范围内
        if rank <= 1:
            top1_correct += 1
        if rank <= 2:
            top2_correct += 1
        if rank <= 3:
            top3_correct += 1

    # 如果没有有效记录，返回提示
    if total_entries == 0:
        print("警告：文件中没有 true 不为 0 的有效记录")
        return None

    # 计算准确率、MR 和 MRR
    result = {
        "Top-1 Accuracy": top1_correct / total_entries,
        "Top-2 Accuracy": top2_correct / total_entries,
        "Top-3 Accuracy": top3_correct / total_entries,
        "MR": rank_sum / total_entries,  # Mean Rank
        "MRR": reciprocal_rank_sum / total_entries,  # Mean Reciprocal Rank
        "Total Entries": total_entries
    }
    return result


# 获取命令行参数（已注释的部分保持不变）
json_file_path = "/home/xxx/code/rcmel0215/data/WikiDiverse/result.json"
# 生成输出文件路径（在原文件名后加 "-metric" 后缀）
output_file_path = json_file_path.replace(".json", "-wikidiverse-top10-metric.txt")

# 计算指标
result = calculate_metrics(json_file_path)

# 如果结果有效，则保存到文件并打印
if result:
    with open(output_file_path, "w") as f:
        f.write(f"总记录数: {result['Total Entries']}\n")
        f.write(f"Top-1 准确率: {result['Top-1 Accuracy']:.6f}\n")
        f.write(f"Top-2 准确率: {result['Top-2 Accuracy']:.6f}\n")
        f.write(f"Top-3 准确率: {result['Top-3 Accuracy']:.6f}\n")
        f.write(f"MR: {result['MR']:.6f}\n")
        f.write(f"MRR: {result['MRR']:.6f}\n")

    print(f"结果已保存到 {output_file_path}")
    # 打印结果
    print(f"总记录数: {result['Total Entries']}")
    print(f"Top-1 acc: {result['Top-1 Accuracy']:.6f}")
    print(f"Top-2 acc: {result['Top-2 Accuracy']:.6f}")
    print(f"Top-3 acc: {result['Top-3 Accuracy']:.6f}")
    print(f"MR: {result['MR']:.6f}")
    print(f"MRR: {result['MRR']:.6f}")