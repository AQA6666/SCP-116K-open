import json
import time
import argparse
from multiprocessing import Pool
from utils import load_jsonl, write_jsonl
from gpt4_request import request_one_turn
import random

def parse_args():
    parser = argparse.ArgumentParser(description='Filter PCB textbooks from library metadata')
    parser.add_argument('--input', type=str, required=True,
                      help='Input JSONL file path containing library metadata')
    parser.add_argument('--output', type=str, required=True,
                      help='Output JSONL file path for filtered PCB textbooks')
    return parser.parse_args()

lib_meta_datas = load_jsonl('lib_meta_info_problems_and_questions.jsonl')
random.shuffle(lib_meta_datas)


def filter_pcb_textbook(meta_data):
    prompt = f'''
Please determine whether the following book belongs to the category of **textbooks or problem books in the fields of physics, chemistry, or biology (including their subfields) targeted at undergraduate to doctoral-level students**. 

If the book is either a textbook or a problem book in these fields, output "Yes". If it does not belong to either category, output "No". 

Consider the information provided carefully and reason through your judgment step by step. Provide your detailed reasoning before delivering the final determination.

Here is the book's metadata:
- **Title**: {meta_data['title']}
- **Author**: {meta_data['author']}

After reasoning, output the answer in the following format:  
[Determine Begin]Yes/No[Determine End]
'''
    max_retries = 2
    retries = 0

    while retries <= max_retries:
        try:
            response = request_one_turn(prompt)
            print(prompt, response)
            print('---')
            response = response[response.find("[Determine Begin]"):]
            if 'Yes' in response:
                meta_data['is_pcb_textbook'] = True
                return meta_data
            else:
                meta_data['is_pcb_textbook'] = False
                return meta_data
        except Exception as e:
            retries += 1
            if retries > max_retries:
                print(f"Error: {e}")
                meta_data['is_pcb_textbook'] = False
                return meta_data
            time.sleep(10)  # 等待10秒后重试

    return meta_data

def main():
    args = parse_args()
    lib_meta_datas = load_jsonl(args.input)
    random.shuffle(lib_meta_datas)

    with Pool() as pool:
        results = pool.map(filter_pcb_textbook, lib_meta_datas)

    lib_meta_datas = [item for item in results if item['is_pcb_textbook']]
    print(len(lib_meta_datas))
    write_jsonl(args.output, lib_meta_datas)

if __name__ == '__main__':
    main()
