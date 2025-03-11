"""
batch_selfinstruct_generate.py

run:
python -m generate_instruction generate_instruction_following_data \
  --output_dir ./ \
  --num_instructions_to_generate 10 \
  --model_name="text-davinci-003" \
"""
import time
import json
import os
import random
import re
import string
from functools import partial
from multiprocessing import Pool
from openai import OpenAI
import numpy as np
import tqdm
from rouge_score import rouge_scorer
import utils
from pathlib import Path
import fire


def encode_prompt(prompt_path, prompt_instructions, num_generated_task_per_request):
    """Encode multiple prompt instructions into a list of messages for chat models."""
    # with open("./stanford_alpaca/prompt_mindat.txt", "r") as f:
    with open(prompt_path, "r") as f:
        system_content = f.read().strip()

    system_message = {"role": "system", "content": system_content}
    messages = [system_message]

    examples_content = "Here are some example training data with their inputs, outputs, and labels:\n\n"
    for idx, task_dict in enumerate(prompt_instructions, 1):
        # instruction = re.sub(r"\s+", " ", task_dict["instruction"]).strip().rstrip(":")
        # instruction_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>    You are a supervisor managing a conversation between the workers: ['GEOMATERIAL_COLLECTOR', 'LOCALITY_COLLECTOR', 'NETWORK_PLOTTER', 'HEATMAP_PLOTTER']. \n Respond with FINISH if the request is fulfilled. \nTeam members description:\n - GEOMATERIAL_COLLECTOR: the geomaterial collector agent, should be called in the first place to obtain mineral dataset\n - LOCALITY_COLLECTOR: the mineral locality collector agent, should be called in the first place to obtain locality dataset\n - NETWORK_PLOTTER: the network visualization plotter agent, will plot the network, cannot plot without the dataset of GEOMATERIAL_COLLECTOR\n - HEATMAP_PLOTTER: the heatmap visualization plotter agent, will plot the heatmap, cannot plot without the dataset of GEOMATERIAL_COLLECTOR\n Please revise your response according to errors if present.\n    <|eot_id|><|start_header_id|>user<|end_header_id|>    Messages : {messages} \n Given the conversation above, who should act next?\n Or should we FINISH? Select one of: ['GEOMATERIAL_COLLECTOR', 'LOCALITY_COLLECTOR', 'NETWORK_PLOTTER', 'HEATMAP_PLOTTER', 'FINISH']\n    <|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        instruction_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>    You are a supervisor managing a conversation between the workers: ['GEOMATERIAL_COLLECTOR', 'LOCALITY_COLLECTOR', 'NETWORK_PLOTTER', 'HEATMAP_PLOTTER']. \n Respond with FINISH if the request is fulfilled.\n  Messages : {messages} \n Given the conversation above, who should act next?\n Or should we FINISH? Select one of: ['GEOMATERIAL_COLLECTOR', 'LOCALITY_COLLECTOR', 'NETWORK_PLOTTER', 'HEATMAP_PLOTTER', 'FINISH']\n    <|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        instruction = instruction_template.format(messages=task_dict["input"])
        # input_text = task_dict["input"]
        output = task_dict["output"]
        label = task_dict["label"]
        
        examples_content += f"{idx}. Instruction: {instruction}\n Output: {output}\n Label: {label}\n"

    examples_message = {"role": "user", "content": examples_content.strip()}
    messages.append(examples_message)

    generate_message = {"role": "user", "content": "Now, please generate {num_task} new diverse task input and output pair following the requirements outlined in the system message. For each pair, provide an appropriate label.".format(num_task = num_generated_task_per_request)}
    messages.append(generate_message)

    return messages

def parse_instruction(raw_instruction):
    pattern = r'.*?({.*})'
    match = re.match(pattern, raw_instruction, re.DOTALL)
    
    if match:
        json_str = match.group(1)
        try:
            data = json.loads(json_str)
            return {
                'input': data['input'],
                'output': data['output'],
                'label': data['label']
            }
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return None
    else:
        print("No valid JSON found in the instruction")
        return None

def post_process_chat_response(num_prompt_instructions, response):
    if response is None:
        print("Response is None")
        return []
    
    # print(f"Raw response: {response.text}")
    print(f"Finish reason: {response.finish_reason}")
    
    raw_instructions = response.text.strip().split('\n\n')
    processed_instructions = []
    
    for raw_instruction in raw_instructions:
        # match = re.match(r"(\d+)\.\s*input:\s*(.*?)\s*output:\s*(.*)label:\s*(.*?)\s*", raw_instruction, re.DOTALL)

        match = parse_instruction(raw_instruction)
        
        if not match:
            print(f"Unable to match instruction and input for: {raw_instruction}")
            continue
        
        # instruction = match.group(2).strip()
        input_text = match['input']
        output = match['output']
        label = match['label']
        
        # input_text = "" if input_text.lower() == "<noinput>" else input_text
        
        # print(f"Extracted instruction: {instruction}")
        # print(f"Extracted input: {input_text}")
        # print(f"Extracted output: {output}")
        
        # Apply filtering rules
        # if len(instruction.split()) <= 3 or len(instruction.split()) > 150:
        #     print("Instruction filtered: too short or too long")
        #     continue
        
        # 其他过滤规则保持不变
        # blacklist = [
        #     "image", "images", "graph", "graphs", "picture", "pictures", "file", "files",
        #     "map", "maps", "draw", "plot", "go to", "video", "audio", "music", "flowchart", "diagram",
        # ]
        # if any(find_word_in_string(word, instruction) for word in blacklist):
        #     print("Instruction filtered: contains blacklisted word")
        #     continue
        
        # if instruction.startswith("Write a program"):
        #     print("Instruction filtered: starts with 'Write a program'")
        #     continue
        
        # if instruction[0] in string.punctuation:
        #     print("Instruction filtered: starts with punctuation")
        #     continue
        
        # if not instruction[0].isascii():
        #     print("Instruction filtered: starts with non-ASCII character")
        #     continue
        
        processed_instructions.append({"input": input_text, "output": output, "label": label})
    
    print(f"Returning instructions: {processed_instructions}")
    return processed_instructions


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def generate_instruction_following_data(
    output_dir="./stanford_alpaca/ward/output",
    # seed_tasks_path="./seed_tasks.jsonl",
    seed_tasks_path="stanford_alpaca/seed_tasks/human_seed_1023.jsonl",
    num_instructions_to_generate=100,
    model_name="gpt-4o-mini",
    num_prompt_instructions=5,
    request_batch_size=1,
    temperature=1.0,
    top_p=1.0,
    num_cpus=16,
    generation_dataset_filename = 'gentask.json',
    num_generated_task_per_request = 5,
    prompt_path = ''
):
    seed_tasks = [json.loads(l) for l in open(seed_tasks_path, "r")]
    seed_instruction_data = [
        {"input": t["input"], "output": t["output"], "label": t["label"]}
        for t in seed_tasks
    ]
    print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")

    os.makedirs(output_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    machine_instruction_data = []
    if os.path.exists(os.path.join(output_dir, generation_dataset_filename)):
        machine_instruction_data = utils.jload(os.path.join(output_dir, generation_dataset_filename))
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

    # similarities = {}
    # scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=num_instructions_to_generate)
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))

    # Comment out the instruction tokenization because we dont need it
    # # first we tokenize all the seed instructions and generated machine instructions
    # all_instructions = [d["instruction"] for d in seed_instruction_data] + [
    #     d["instruction"] for d in machine_instruction_data
    # ]
    # all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]

    # client = OpenAI()  # Initialize OpenAI client

    while len(machine_instruction_data) < num_instructions_to_generate:
        request_idx += 1

        prompt_instructions = random.sample(seed_instruction_data, num_prompt_instructions)
        messages = encode_prompt(prompt_path, prompt_instructions, num_generated_task_per_request)
        
        decoding_args = utils.OpenAIDecodingArguments(
            temperature=temperature,
            n=1,
            max_tokens=3072,
            top_p=top_p,
            stop=["\n4.", "4.", "4."],
        )
        
        request_start = time.time()
        results = utils.openai_completion(
            prompts=[messages], 
            model_name=model_name,
            batch_size=1,
            decoding_args=decoding_args,
        )
        request_duration = time.time() - request_start
        
        print(f"Number of results received: {len(results)}")
        
        process_start = time.time()
        for result in results:
            # print(f"Processing result: {result}")
            new_instructions = post_process_chat_response(num_prompt_instructions, result)
            # print(f"New instructions generated: {new_instructions}")
            
            for instruction_data in new_instructions:
                machine_instruction_data.append(instruction_data)
                # comment out the instruction tokenization
                # all_instructions.append(instruction_data["instruction"])
                # all_instruction_tokens.append(scorer._tokenizer.tokenize(instruction_data["instruction"]))
                progress_bar.update(1)

        process_duration = time.time() - process_start
        print(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
        print(f"Current total instructions: {len(machine_instruction_data)}")
        
        utils.jdump(machine_instruction_data, os.path.join(output_dir, generation_dataset_filename))

def wrapped_execution_fn(**kwargs):
    generate_instruction_following_data(
        prompt_path=kwargs.get('PROMPT_PATH'),
        seed_tasks_path = kwargs.get('SAMPLE_DATASET_PATH'),
        output_dir = kwargs.get('OUTPUT_PATH'),
        generation_dataset_filename=kwargs.get('OUTPUT_FILENAME'),
        num_instructions_to_generate=kwargs.get('num_instructions_to_generate', 10),
        num_generated_task_per_request=kwargs.get('num_generated_task_per_request', 10)
    )
    return


def main(task, **kwargs):
    globals()[task](**kwargs)


if __name__ == "__main__":
    # fire.Fire(main)

    prompt_filename = 'prompt_claude_v3.txt'
    sample_filename = 'human_seed_1023.jsonl'

    prompt_path = Path('prompt', prompt_filename)
    sample_dataset_path = Path('sample', sample_filename)
    output_path = 'output'
    output_filename = 'output_generated_tasks.json'
    
    

    wrapped_execution_fn(PROMPT_PATH = prompt_path,
                        SAMPLE_DATASET_PATH = sample_dataset_path,
                        OUTPUT_PATH = output_path,
                        OUTPUT_FILENAME = output_filename,
                        num_instructions_to_generate = 2000,
                        num_generated_task_per_request = 10
                    )


    # generate_instruction_following_data(num_instructions_to_generate=10, generation_dataset_filename=filename,
    # num_generated_task_per_request = 10)