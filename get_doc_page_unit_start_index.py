from utils import load_jsonl, write_jsonl, find_files
from gpt4_request import safe_request_one_turn
import random
from multiprocessing import Pool
import json
from tqdm import tqdm
import argparse


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

Please ignore the following:
    1. Headers and footers (especially on line 0, 1, 2)
    2. Sub-question markers like "(1)", "(a)", "(i)", etc.
    3. Solution indicators such as "**SOLUTION:**", "## Solution", "### General Solution", etc.

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
    response = safe_request_one_turn(prompt, error_response="```json\n[]\n```", model="gpt-4o")
    print(prompt, response)
    print('-'*10)
    try:
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


# Add argument parser before the main execution
parser = argparse.ArgumentParser(description='Process book pages to identify unit start indices')
parser.add_argument('source_file_dir', type=str, help='Directory containing the book JSONL files')

args = parser.parse_args()

# Replace the hardcoded path with the command line argument
book_paths = find_files(args.source_file_dir, '*.jsonl')

for book_path in tqdm(book_paths):
    print(book_path)
    book_pages = load_jsonl(book_path)
    with Pool() as pool:
        book_pages = pool.map(get_unit_start_index, book_pages)
    write_jsonl(book_path, book_pages)

