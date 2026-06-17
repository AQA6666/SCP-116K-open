from utils import load_jsonl, write_jsonl, find_files
import openai
import random
from multiprocessing import Pool
import json
from tqdm import tqdm
import argparse
import os


def add_line_index_to_page(page_content):
    lines = page_content.split('\n')
    page_with_line_index = ''
    for i, line in enumerate(lines):
        page_with_line_index += f'{i}|{line}\n'
    return page_with_line_index


def get_unit_start_index(page):
    page_text_with_line_index = add_line_index_to_page(page['text'])
    prompt = f'''For the given book page:
---
{page_text_with_line_index}
---
Please identify if there are any:
    1. Chapter beginnings
    2. Section beginnings
    3. Subsection beginnings
    4. Problem (exercise or example) beginnings
    5. Independent solution (that means the solution of a problem without problem before it and with solution number) beginnings

Please ignore the following:
    1. Headers and footers (especially on line 0, 1)
    2. Sub-question markers like "(1)", "(a)", "(i)", etc.
    3. Solution indicators without solution number such as "**SOLUTION:**", "## Solution", "### General Solution", etc.

Let's solve this step by step:
    1. identify any chapter indicators (e.g., "Chapter 1", "第一章", etc.)
    2. look for section markers (e.g., "1.1", "Section 1", etc.)
    3. identify subsection markers (e.g., "1.1.1", etc.)
    4. look for problem indicators (e.g., "1.1", "1-1", "**1008**", "Exercise 1", "Problem 1", "Example 1.1", "习题1.", "例题1.", etc.)
    5. For each identified element: Check if it's a start of a chapter/section/subsection/problem and **it's not part of the elements to be ignored as specified above**

First, explain your reasoning process strictly following the 1~5 steps above. 
Then, provide the list of line numbers in JSON format, for example:
```json
[1, 2, 3]
```
'''
    try:
        client = openai.OpenAI(
            base_url=args.base_url,
            api_key=args.api_key,
        )
        response = client.chat.completions.create(
            model=args.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=args.temperature
        ).choices[0].message.content
        print(prompt, response)
        print('-'*10)
        response = response.split('```json')[-1].split('```')[0]
        response = json.loads(response)
        print('json load success', json.dumps(response, ensure_ascii=False))
    except Exception as e:
        print('*******\n' * 6)
        print(e)
        print('*******\n' * 6)
        response = []
    page['unit_start_index'] = response
    return page
    # return page_text_with_line_index + '\n[SPLIT]\n' + str(page['id']) + '\n[SPLIT]\n' + json.dumps(response, ensure_ascii=False)

def sanitize_filename(filename):
    # Extract directory path and base filename
    directory = os.path.dirname(filename)
    basename = os.path.basename(filename)
    
    # Replace invalid characters with underscores in the basename only
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        basename = basename.replace(char, '_')
    
    # Recombine directory and sanitized basename
    return os.path.join(directory, basename)

args = argparse.ArgumentParser()
args.add_argument("--data_path", type=str, default="../zh")
args.add_argument("--model_name", type=str, default="gpt-4o")
args.add_argument("--process_num", type=int, default=10)
args.add_argument("--base_url", type=str, default="http://openai.infly.tech/v1/")
args.add_argument("--api_key", type=str, default="dummy")
args.add_argument("--temperature", type=float, default=1.0)
args = args.parse_args()

book_paths = find_files(args.data_path, '*.jsonl')
# book_paths = random.sample(book_paths, 5)

for book_path in tqdm(book_paths):
    print(book_path)
    book_pages = load_jsonl(book_path)
    # book_page = random.choice(book_pages)
    with Pool() as pool:
        book_pages = pool.map(get_unit_start_index, book_pages)
    # book_page = get_unit_start_index(book_page)
    write_jsonl(book_path, book_pages)
    # safe_filename = sanitize_filename(book_path[:-6] + '_unit_start_index.txt')
    # print(safe_filename)
    # with open(safe_filename, 'w', encoding='utf-8') as f:
    #     f.write(book_page)
