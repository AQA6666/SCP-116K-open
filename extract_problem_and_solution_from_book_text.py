import json
import random
import time
from multiprocessing import Pool
import tiktoken
from tqdm import tqdm
from gpt4_request import request_one_turn
from utils import load_jsonl, write_jsonl, find_files
import argparse


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
            response = request_one_turn(prompt, temperature=0.4, top_p=0.5)
            print(response)
            chunk['response'] = response
            if 'None' in response:
                return chunk
            response = response.split('```json')[1].split('```')[0]
            response = response.replace("\\", "\\\\")
            response = json.loads(response)
            chunk['problems_and_solutions'] = response
            return chunk
        except Exception as e:
            retries += 1
            if retries > max_retries:
                print(f"Error: {e}")
                chunk['error'] = {'response': response, 'error': str(e)}
                return chunk
            time.sleep(10)
    return chunk




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract problems and solutions from book text chunks')
    parser.add_argument('--input', type=str, required=True,
                      help='Input JSONL file path containing book chunks')
    parser.add_argument('--output', type=str, required=True,
                      help='Output JSONL file path for storing problems and solutions')
    
    args = parser.parse_args()
    
    chunks = load_jsonl(args.input)
    print(len(chunks))
    for i in range(0, len(chunks)):
        chunks[i]['chunk_number'] = i
    with Pool() as pool:
        results = pool.map(extract_problem_and_solutions, chunks)

    write_jsonl(args.output, results)
