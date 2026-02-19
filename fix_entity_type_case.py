import json

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, desc=None):
        return iterable


def fix_entity_type_case(input_path: str, output_path: str = None):
    """
    修复 JSON 文件中所有实体类型的大小写，统一转换为小写
    
    Args:
        input_path: 输入的 JSON 文件路径
        output_path: 输出的 JSON 文件路径，如果为 None 则覆盖原文件
    """
    if output_path is None:
        output_path = input_path
    
    print(f"Loading data from {input_path}...")
    
    # 读取 JSON 文件
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples.")
    print("Converting entity types to lowercase...")
    
    # 修复每个样本
    fixed_count = 0
    total_samples = len(data)
    for i, sample in enumerate(tqdm(data, desc="Processing")):
        entities = sample.get('entities', [])
        types = sample.get('types', [])
        
        # 修复每个实体的类型
        for entity in entities:
            if 'type' in entity:
                original_type = entity['type']
                entity['type'] = original_type.lower()
                if original_type != entity['type']:
                    fixed_count += 1
        
        # 修复 types 列表
        if types:
            sample['types'] = [t.lower() for t in types]
        
        # 如果没有 tqdm，每处理1000个样本打印一次进度
        if not HAS_TQDM and (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{total_samples} samples...")
    
    print(f"Fixed {fixed_count} entity types.")
    print(f"Saving to {output_path}...")
    
    # 保存修复后的数据
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    input_path = "./dataset/instruct_uie_ner_converted.json"
    output_path = "./dataset/instruct_uie_ner_converted.json"  # 覆盖原文件
    
    fix_entity_type_case(input_path, output_path)
