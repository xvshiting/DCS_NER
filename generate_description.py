import json
import os
import dashscope
from tqdm import tqdm   
import argparse

system_prompt = """/no_think
You generate extraction-oriented label descriptions for multi-label NER.
Descriptions are used as prompts for span extraction models.
Do not include entity strings, examples, or quotes from the text.
Do not explain your reasoning.
Follow the output format strictly."""

gen_prompt = """Text:
{TEXT}

The following entity types appear in the text.
Each entity type may have multiple mentions.

For each entity type, write SIX short definitions that help extract its mentions.

Requirements:
- Definitions 1–5 MUST be grounded in THIS text.
- Definition 6 MUST be general and NOT depend on this text.
- All definitions for the same entity type must be semantically consistent.
- Focus on how the entity type is expressed in text, not on real-world ontology.
- Do NOT include any entity names or concrete examples.
- Keep each definition concise (20–45 tokens, Definition 6 can be 15–30 tokens).

Entity types and their mentions:
{entity_label_mentions}

Output format (strict JSON, one object per entity type):

""" 
post_fix_prompt = json.dumps([
  {
    "label": "<label_name>",
    "definitions": {
      "D1": "<canonical, extraction-oriented, context in this Text>",
      "D2": "<paraphrase of D1>",
      "D3": "<pattern-focused, how it appears in this text, true label should not appear.>",
      "D4": "<boundary-focused, include/exclude cues>",
      "D5": "<minimal, compressed version>",
      "D6": "<general definition, text-independent>"
    }
  }
], indent=4)

def gen_entity_mentions_str(data):
    entities = data["entities"]
    entity_type_dict =dict()
    for item in entities:
        name = item["name"]
        _type = item["type"]
        name_list = entity_type_dict.get(_type,[])
        name_list.append(name)
        entity_type_dict[_type] = name_list 
    lines = []
    for k,v in entity_type_dict.items():
        lines.append("{}:[{}]".format(k, ",".join(list(set(v)))))
    return "\n".join(lines)

def generate_prompt(data):
    entity_mention_str  = gen_entity_mentions_str(data)
    # print(type(entity_mention_str))
    prompt = gen_prompt.format(TEXT = data["sentence"], entity_label_mentions=entity_mention_str)
    return prompt+post_fix_prompt


def generate_description(data):
    prompt = generate_prompt(data)
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt}
    ]
    response = dashscope.Generation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
        api_key=os.getenv('DASHSCOPE_API_KEY'),
        model="qwen-plus", # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=messages,
        result_format='message'
        )
    return response["output"]["choices"][0]["message"]["content"]

def main(args):
    input_path = args.input_path
    output_path = args.output_path  #jsonl 
    save_interval = args.save_interval
    start_index = args.start_index
    end_index = args.end_index
    data = json.load(open(input_path, "rb"))
    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = len(data)
    # if has start and end index, output path should add start_index and end_index
    if start_index is not None or end_index is not None:
        output_path = output_path.replace(".jsonl", f"_{start_index}_{end_index}.jsonl")
    data = data[start_index:end_index]
    with open(output_path, "w") as f:
        #tqdm progress bar
        for item in tqdm(data, desc="Generating description"):
            description = generate_description(item)
            loaded_description = json.loads(description)
            item["description"] = loaded_description
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            if len(data) % save_interval == 0:
                f.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./dataset/instruct_uie_ner_converted.json")
    parser.add_argument("--output_path", type=str, default="./dataset/instruct_uie_ner_converted_description.jsonl")
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--start_index", type=int, default=None)
    parser.add_argument("--end_index", type=int, default=None)  
    # pool size
    args = parser.parse_args()
    main(args)