import json
import re
import ast
from typing import List, Dict, Any, Optional
from datasets import load_dataset


def find_entity_positions(text: str, entity: str) -> List[tuple]:
    """
    在文本中找到实体的所有位置（支持大小写不敏感匹配）
    返回 [(start, end), ...] 列表
    """
    positions = []
    entity_lower = entity.lower()
    text_lower = text.lower()
    
    # 转义特殊字符
    entity_escaped = re.escape(entity)
    
    # 尝试精确匹配（大小写不敏感）
    pattern = re.compile(re.escape(entity), re.IGNORECASE)
    for match in pattern.finditer(text):
        start, end = match.span()
        positions.append((start, end))
    
    return positions


def extract_label_from_question(question: str) -> str:
    """
    从问题中提取标签并转换为小写
    例如: "What describes Measurement in the text?" -> "measurement"
    """
    # 匹配 "What describes [Label] in the text?"
    match = re.search(r"What describes (.+?) in the text\?", question, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()  # 转换为小写
    return ""


def parse_entities_from_answer(answer: str) -> List[str]:
    """
    从答案中解析实体列表
    例如: '["entity1", "entity2"]' -> ["entity1", "entity2"]
    """
    try:
        # 使用 ast.literal_eval 安全地解析列表
        entities = ast.literal_eval(answer)
        if isinstance(entities, list):
            return [str(e).strip() for e in entities if e]
        return []
    except (ValueError, SyntaxError):
        return []


def convert_pile_to_instruct_uie(pile_item: Dict[str, Any], dataset_name: str = "Pile_NER_type") -> Optional[Dict[str, Any]]:
    """
    将 pile ner type 格式转换为 instruct_uie_ner 格式
    
    Args:
        pile_item: pile ner type 格式的数据项
        dataset_name: 数据集名称，默认为 "Pile_NER_type"
    
    Returns:
        instruct_uie_ner 格式的数据项，如果转换失败则返回 None
    """
    conversations = pile_item.get('conversations', [])
    
    # 1. 提取文本（第一个 human 消息，去掉 "Text: " 前缀）
    text = ""
    for conv in conversations:
        if conv.get('from') == 'human' and conv.get('value', '').startswith('Text:'):
            text = conv['value'].replace('Text:', '', 1).strip()
            break
    
    if not text:
        return None
    
    # 2. 提取实体和标签
    entities = []
    types_set = set()
    
    i = 0
    while i < len(conversations):
        conv = conversations[i]
        
        # 找到问题（human 消息，包含 "What describes"）
        if conv.get('from') == 'human' and 'What describes' in conv.get('value', ''):
            question = conv['value']
            label = extract_label_from_question(question)
            
            # 找到对应的答案（下一个 gpt 消息）
            if i + 1 < len(conversations) and conversations[i + 1].get('from') == 'gpt':
                answer = conversations[i + 1]['value']
                entity_names = parse_entities_from_answer(answer)
                
                # 为每个实体找到位置并添加到 entities 列表
                # 确保 label 是小写
                label_lower = label.lower() if label else ""
                for entity_name in entity_names:
                    if entity_name:  # 跳过空实体
                        positions = find_entity_positions(text, entity_name)
                        if positions:
                            # 为每个找到的位置创建一个实体
                            for start, end in positions:
                                entities.append({
                                    'name': text[start:end],  # 使用原始文本中的实际字符串
                                    'pos': [start, end],
                                    'type': label_lower  # 使用小写的标签
                                })
                            if label_lower:
                                types_set.add(label_lower)
                i += 2  # 跳过问题和答案
                continue
        
        i += 1
    
    # 3. 构建 instruct_uie_ner 格式
    result = {
        'sentence': text,
        'entities': entities,
        'dataset': dataset_name,
        'types': sorted(list(types_set))
    }
    
    return result


def convert_dataset(pile_dataset_path: str, output_path: str = None, split: str = "train"):
    """
    转换整个数据集
    
    Args:
        pile_dataset_path: pile ner type 数据集路径
        output_path: 输出路径（可选，如果为 None 则返回转换后的数据）
        split: 数据集分割名称，默认为 "train"
    """
    # 加载数据集
    print(f"Loading dataset from {pile_dataset_path}...")
    dataset = load_dataset(pile_dataset_path)
    
    if split not in dataset:
        print(f"Warning: split '{split}' not found. Available splits: {list(dataset.keys())}")
        split = list(dataset.keys())[0]
        print(f"Using split '{split}' instead.")
    
    # 转换数据
    print(f"Converting {len(dataset[split])} examples...")
    converted_data = []
    
    for i, item in enumerate(dataset[split]):
        converted_item = convert_pile_to_instruct_uie(item)
        if converted_item:
            converted_data.append(converted_item)
        
        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{len(dataset[split])} examples...")
    
    print(f"Conversion completed! Converted {len(converted_data)} examples.")
    
    # 保存或返回
    if output_path:
        print(f"Saving to {output_path}...")
        # 保存为 JSON 文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, ensure_ascii=False, indent=2)
        print(f"Saved to {output_path}")
    else:
        return converted_data


if __name__ == "__main__":
    # 使用示例
    pile_path = "/data/dataset/ner/Pile_NER_type"
    output_path = "./dataset/instruct_uie_ner_converted.json"
    
    # 转换数据集
    convert_dataset(pile_path, output_path, split="train")
    
    # 或者只转换单个样本进行测试
    # dataset = load_dataset(pile_path)
    # sample = dataset["train"][0]
    # converted = convert_pile_to_instruct_uie(sample)
    # print(json.dumps(converted, indent=2, ensure_ascii=False))
