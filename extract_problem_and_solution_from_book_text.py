import json
import random
import time
from multiprocessing import Pool
import tiktoken
from tqdm import tqdm
from openai import OpenAI
from utils import load_jsonl, write_jsonl, find_files
import json
import argparse


args = argparse.ArgumentParser()
args.add_argument("--data_path", type=str, default="math_book_chunks/math_books_609_chunks_2000_4000.jsonl")
args.add_argument("--model_name", type=str, default="gpt-4o")
args.add_argument("--process_num", type=int, default=10)
args.add_argument("--base_url", type=str, default="http://openai.infly.tech/v1/")
args.add_argument("--api_key", type=str, default="dummy")
args.add_argument("--temperature", type=float, default=0.4)
args = args.parse_args()

# print the args in a readable format
print(f"Model name: {args.model_name}")
print(f"Process number: {args.process_num}")
print(f"Base URL: {args.base_url}")
print(f"API key: {args.api_key}")
print(f"Temperature: {args.temperature}")


def extract_problem_and_solutions(chunk):
    print(chunk['chunk_number'])
    prompt = f'''
Input:
------
{chunk['chunk']}
------

I am a university professor preparing an exercise problem bank. 

Please help me extract the problems (include examples) or solutions from provided textbook pages.

1. First, find all the problems or solutions in the provided content. *Carefully analyze each piece of content to determine whether it is a problem or a solution.* 
2. Ensure each identified problem is complete and not part of a solution or other content.
3. *For problems with multiple sub-problems, DO NOT omit the problem statement, DO NOT split the problem with multiple sub-problems.*
4. *DO NOT omit or change any part of the problems and solutions. Ensure the content is complete.*

Output the extracted data as a list of JSON objects.

Let's think step by step, output your thought process, and then output the extracted results in the following format:

```json
[
    {{
        "problem number": "problem number in book, such as 1.1",
        "problem": "Full content of problem 1.1 .",
    }},
    {{
        "solution number": "1.1",
        "solution": "Full content of solution 1.1 .",
    }}
    {{
        "problem number": "1.2",
        "problem": "Full content of problem 1.2 .",
    }}
]
```
If no problems and solutions are present in the provided content, output an empty list:
```json
[]
```
This task is important for my work, so please strictly follow the requirements.
'''
    max_retries = 2
    retries = 0
    response = ''
    while retries <= max_retries:
        try:
            client = OpenAI(api_key=args.api_key, base_url=args.base_url)
            if 'gpt' in args.model_name or 'o' in args.model_name:
                response = client.chat.completions.create(
                    model=args.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    extra_body={},
                    stream=False,
                    temperature=args.temperature
                )
                
                response_content = response.choices[0].message.content
            else:
                response = client.chat.completions.create(
                    model=args.model_name, 
                    messages=[{'role': 'user', 'content': prompt}],
                    temperature=args.temperature,
                    stream=False
                )
                
                response_content = response.choices[0].message.content
            
            print(response_content)
            # chunk['response'] = response_content
            json_content = response_content.split('```json')[-1].split('```')[0]
            json_content = json_content.replace("\\", "\\\\")
            json_content = json.loads(json_content)
            chunk['problems_and_solutions'] = json_content
            return chunk
        except Exception as e:
            retries += 1
            if retries > max_retries:
                print(f"Error: {e}")
                chunk['error'] = {'response': response_content if 'response_content' in locals() else response, 'error': str(e)}
                return chunk
            time.sleep(30)
    return chunk



data = load_jsonl(args.data_path)
# data = load_jsonl("../book_chunks/math_books_609_chunks_2000_4000.jsonl") + load_jsonl("../book_chunks/pcb_books_467_chunks_2000_4000.jsonl") + load_jsonl("../book_chunks/zh_xiti_math_book_chunk.jsonl") + load_jsonl("../book_chunks/zh_xiti_sci_book_chunk.jsonl")
total_num = len(data)
# data = random.sample(data, 100)
# 为data中的每个元素添加chunk_number
for i, chunk in enumerate(data):
    chunk['chunk_number'] = i

with Pool(args.process_num) as pool:
    results = pool.map(extract_problem_and_solutions, data)

write_jsonl(args.data_path.replace('.jsonl', f'_problems_and_solutions_extracted_by_{args.model_name}.jsonl'), results)

