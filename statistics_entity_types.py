import json
from collections import Counter, defaultdict
from typing import Dict, List, Set


def load_and_statistics_entity_types(json_path: str, output_path: str = "entity_type_statistics.txt"):
    """
    加载 instruct_uie_ner_converted.json 文件，统计每个实体类型的数量和对应的文档数量，
    并按照从高到低排序输出到txt文件
    
    Args:
        json_path: JSON文件路径
        output_path: 输出txt文件路径
    """
    print(f"Loading data from {json_path}...")
    
    # 统计变量
    entity_type_counter = Counter()  # 实体类型 -> 实体数量
    entity_type_docs = defaultdict(set)  # 实体类型 -> 文档索引集合
    total_entities = 0
    total_samples = 0
    samples_with_entities = 0
    samples_without_entities = 0
    
    # 读取JSON文件
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} samples.")
        
        # 遍历每个样本
        for doc_idx, sample in enumerate(data):
            total_samples += 1
            entities = sample.get('entities', [])
            
            if entities:
                samples_with_entities += 1
                # 统计每个实体类型
                doc_types = set()  # 记录这个文档中出现的所有类型
                for entity in entities:
                    entity_type = entity.get('type', '')
                    if entity_type:
                        entity_type_counter[entity_type] += 1
                        total_entities += 1
                        doc_types.add(entity_type)
                
                # 记录每个类型出现在哪些文档中
                for entity_type in doc_types:
                    entity_type_docs[entity_type].add(doc_idx)
            else:
                samples_without_entities += 1
        
    except FileNotFoundError:
        print(f"Error: File {json_path} not found!")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON file: {e}")
        return
    
    # 按照实体数量从高到低排序
    sorted_types_by_count = sorted(entity_type_counter.items(), key=lambda x: x[1], reverse=True)
    
    # 按照文档数量从高到低排序
    sorted_types_by_docs = sorted(
        [(etype, len(entity_type_docs[etype])) for etype in entity_type_counter.keys()],
        key=lambda x: x[1],
        reverse=True
    )
    
    # 写入结果到文件
    print(f"Writing statistics to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        # 写入总体分布描述
        f.write("=" * 80 + "\n")
        f.write("实体类型统计报告\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("【总体分布描述】\n")
        f.write("-" * 80 + "\n")
        f.write(f"总样本数: {total_samples:,}\n")
        f.write(f"包含实体的样本数: {samples_with_entities:,}\n")
        f.write(f"不包含实体的样本数: {samples_without_entities:,}\n")
        f.write(f"总实体数: {total_entities:,}\n")
        f.write(f"唯一实体类型数: {len(entity_type_counter):,}\n")
        f.write(f"平均每个样本的实体数: {total_entities / total_samples if total_samples > 0 else 0:.2f}\n")
        f.write(f"平均每个有实体样本的实体数: {total_entities / samples_with_entities if samples_with_entities > 0 else 0:.2f}\n")
        f.write("\n")
        
        # 写入实体类型统计（按文档数量从高到低排序 - 主要输出）
        f.write("=" * 80 + "\n")
        f.write("实体类型详细统计（按文档数量从高到低排序）\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'排名':<6} {'实体类型':<45} {'文档数量':<12} {'实体数量':<12} {'文档占比':<10} {'实体占比':<10}\n")
        f.write("-" * 80 + "\n")
        
        for rank, (entity_type, doc_count) in enumerate(sorted_types_by_docs, start=1):
            entity_count = entity_type_counter[entity_type]
            doc_percentage = (doc_count / samples_with_entities * 100) if samples_with_entities > 0 else 0
            entity_percentage = (entity_count / total_entities * 100) if total_entities > 0 else 0
            f.write(f"{rank:<6} {entity_type:<45} {doc_count:<12,} {entity_count:<12,} {doc_percentage:>6.2f}% {entity_percentage:>6.2f}%\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        f.write("实体类型详细统计（按实体数量从高到低排序）\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'排名':<6} {'实体类型':<45} {'实体数量':<12} {'文档数量':<12} {'实体占比':<10} {'文档占比':<10}\n")
        f.write("-" * 80 + "\n")
        
        for rank, (entity_type, entity_count) in enumerate(sorted_types_by_count, start=1):
            doc_count = len(entity_type_docs[entity_type])
            entity_percentage = (entity_count / total_entities * 100) if total_entities > 0 else 0
            doc_percentage = (doc_count / samples_with_entities * 100) if samples_with_entities > 0 else 0
            f.write(f"{rank:<6} {entity_type:<45} {entity_count:<12,} {doc_count:<12,} {entity_percentage:>6.2f}% {doc_percentage:>6.2f}%\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
        
        # 写入一些额外的统计信息
        f.write("\n【额外统计信息】\n")
        f.write("-" * 80 + "\n")
        if sorted_types_by_count:
            f.write(f"实体数量最多的类型: {sorted_types_by_count[0][0]} (出现 {sorted_types_by_count[0][1]:,} 次)\n")
            f.write(f"实体数量最少的类型: {sorted_types_by_count[-1][0]} (出现 {sorted_types_by_count[-1][1]:,} 次)\n")
            f.write(f"最多与最少实体数量的比例: {sorted_types_by_count[0][1] / sorted_types_by_count[-1][1] if sorted_types_by_count[-1][1] > 0 else 0:.2f}:1\n")
        
        if sorted_types_by_docs:
            f.write(f"\n文档数量最多的类型: {sorted_types_by_docs[0][0]} (出现在 {sorted_types_by_docs[0][1]:,} 个文档中)\n")
            f.write(f"文档数量最少的类型: {sorted_types_by_docs[-1][0]} (出现在 {sorted_types_by_docs[-1][1]:,} 个文档中)\n")
            f.write(f"最多与最少文档数量的比例: {sorted_types_by_docs[0][1] / sorted_types_by_docs[-1][1] if sorted_types_by_docs[-1][1] > 0 else 0:.2f}:1\n")
        
        # 统计分布情况
        if sorted_types_by_count:
            # 前10%的实体类型占比
            top_10_percent_count = len(sorted_types_by_count) // 10 if len(sorted_types_by_count) >= 10 else 1
            top_10_percent_entities = sum(count for _, count in sorted_types_by_count[:top_10_percent_count])
            top_10_percent_ratio = (top_10_percent_entities / total_entities * 100) if total_entities > 0 else 0
            f.write(f"\n前10%的实体类型（{top_10_percent_count}种）占总实体数的比例: {top_10_percent_ratio:.2f}%\n")
            
            # 前50%的实体类型占比
            top_50_percent_count = len(sorted_types_by_count) // 2
            top_50_percent_entities = sum(count for _, count in sorted_types_by_count[:top_50_percent_count])
            top_50_percent_ratio = (top_50_percent_entities / total_entities * 100) if total_entities > 0 else 0
            f.write(f"前50%的实体类型（{top_50_percent_count}种）占总实体数的比例: {top_50_percent_ratio:.2f}%\n")
    
    print(f"Statistics saved to {output_path}")
    print(f"\nSummary:")
    print(f"  - Total samples: {total_samples:,}")
    print(f"  - Total entities: {total_entities:,}")
    print(f"  - Unique entity types: {len(entity_type_counter):,}")
    if sorted_types_by_docs:
        print(f"  - Most common type (by doc count): {sorted_types_by_docs[0][0]} ({sorted_types_by_docs[0][1]:,} documents)")
    if sorted_types_by_count:
        print(f"  - Most common type (by entity count): {sorted_types_by_count[0][0]} ({sorted_types_by_count[0][1]:,} entities)")


if __name__ == "__main__":
    json_path = "./dataset/instruct_uie_ner_converted.json"
    output_path = "./dataset/entity_type_statistics.txt"
    
    load_and_statistics_entity_types(json_path, output_path)
