import json
import os
import dashscope
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    entity_type_dict = dict()
    for item in entities:
        name = item["name"]
        _type = item["type"]
        name_list = entity_type_dict.get(_type, [])
        name_list.append(name)
        entity_type_dict[_type] = name_list
    lines = []
    for k, v in entity_type_dict.items():
        lines.append("{}:[{}]".format(k, ",".join(list(set(v)))))
    return "\n".join(lines)

def generate_prompt(data):
    entity_mention_str = gen_entity_mentions_str(data)
    prompt = gen_prompt.format(TEXT=data["sentence"], entity_label_mentions=entity_mention_str)
    return prompt + post_fix_prompt


def generate_description(data):
    prompt = generate_prompt(data)
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt}
    ]
    response = dashscope.Generation.call(
        api_key=os.getenv('DASHSCOPE_API_KEY', "sk-b95c066723574c7ba72147797a568395"),
        model="qwen-plus",
        messages=messages,
        result_format='message'
    )
    return response["output"]["choices"][0]["message"]["content"]


def worker(chunk_data, global_start, global_end, output_path, save_interval):
    """Each worker processes its own chunk and writes to its own file."""
    output_file = output_path.replace(".jsonl", f"_{global_start}_{global_end}.jsonl")
    with open(output_file, "w") as f:
        for i, item in enumerate(tqdm(chunk_data, desc=f"Worker [{global_start}-{global_end}]")):
            try:
                description = generate_description(item)
                loaded_description = json.loads(description)
                item["description"] = loaded_description
            except Exception as e:
                print(f"[{global_start}-{global_end}] Error at index {global_start + i}: {e}")
                item["description"] = None
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            if (i + 1) % save_interval == 0:
                f.flush()
    print(f"Worker [{global_start}-{global_end}] done, wrote {len(chunk_data)} items to {output_file}")
    return output_file


def multi_process_run(args):
    data = json.load(open(args.input_path, "r", encoding="utf-8"))
    start_index = args.start_index if args.start_index is not None else 0
    end_index = args.end_index if args.end_index is not None else len(data)
    data = data[start_index:end_index]
    print(f"Total items to process: {len(data)} (global index {start_index} to {end_index})")

    pool_size = min(args.pool_size, len(data))
    if pool_size <= 1:
        output_file = args.output_path.replace(".jsonl", f"_{start_index}_{end_index}.jsonl")
        worker(data, start_index, end_index, args.output_path, args.save_interval)
        return

    # Split data into chunks and submit to thread pool
    futures = []
    with ThreadPoolExecutor(max_workers=pool_size) as executor:
        for i in range(pool_size):
            chunk_start = i * len(data) // pool_size
            chunk_end = (i + 1) * len(data) // pool_size
            chunk_data = data[chunk_start:chunk_end]
            # Global indices for file naming
            global_start = start_index + chunk_start
            global_end = start_index + chunk_end
            future = executor.submit(
                worker, chunk_data, global_start, global_end,
                args.output_path, args.save_interval
            )
            futures.append(future)

        # Wait and collect results, re-raise any exceptions
        for future in as_completed(futures):
            try:
                result = future.result()
                print(f"Finished: {result}")
            except Exception as e:
                print(f"Worker failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="./dataset/instruct_uie_ner_converted.json")
    parser.add_argument("--output_path", type=str, default="./dataset/instruct_uie_ner_converted_description.jsonl")
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--start_index", type=int, default=None)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--pool_size", type=int, default=10)
    args = parser.parse_args()
    multi_process_run(args)
