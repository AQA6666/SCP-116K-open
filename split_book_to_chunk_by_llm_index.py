import tiktoken
from tqdm import tqdm
from utils import load_jsonl, write_jsonl, find_files
import random
import argparse

# Add argument parser
parser = argparse.ArgumentParser(description='Split book into chunks')
parser.add_argument('--source_file_dir', type=str, required=True,
                   help='Directory containing source JSONL files')
parser.add_argument('--save_path', type=str, required=True,
                   help='Path to save the output JSONL file')
args = parser.parse_args()

# 判断line是否是sub-problem的开始
def is_sub_problem_start(line):
    line = line.strip()
    sub_problem_start_keywords = ['A.', 'B.', 'C.', 'D.',
                                'a.', 'b.', 'c.', 'd.',
                                '(A)', '(B)', '(C)', '(D)',
                                'A)', 'B)', 'C)', 'D)',
                                '(a)', '(b)', '(c)', '(d)',
                                'a)', 'b)', 'c)', 'd)',
                                '(1)', '(2)', '(3)', '(4)',
                                '1)', '2)', '3)', '4)',
                                '(i)', '(ii)', '(iii)', '(iv)',
                                'i)', 'ii)', 'iii)', 'iv)']
    for keyword in sub_problem_start_keywords:
        if line.startswith(keyword):
            return True
    return False

book_paths = find_files(args.source_file_dir, '*.jsonl')

encoding = tiktoken.get_encoding('cl100k_base')

add_page = True
add_row_number = False

chunk_expected_token_num = 2000

chunks = []
for book_path in tqdm(book_paths):
    book = load_jsonl(book_path)
    current_chunk = ''
    current_chunk_token_num = 0
    current_page_number_list = []
    for i, page in enumerate(book):
        current_chunk += f'\n{f"------page{i}------" if add_page else ""}'
        lines = page['text'].split('\n')
        for j, line in enumerate(lines):
            current_line_token_num = len(encoding.encode(line))
            # 根据line的类型设置split_threshold避免误切的同时保证chunk的token数不会过大
            if j in page['unit_start_index'] and j != 0 and not is_sub_problem_start(line):
                split_threshold = chunk_expected_token_num
            elif j in page['unit_start_index'] and j != 0 and is_sub_problem_start(line):
                split_threshold = 1.5 * chunk_expected_token_num
            elif not line:
                split_threshold = 2 * chunk_expected_token_num
            else:
                split_threshold = 4 * chunk_expected_token_num  

            # 如果需要创建新的chunk
            if current_chunk_token_num + current_line_token_num > split_threshold:
                chunks.append({
                    'book': book_path,
                    'page_number_list': current_page_number_list,
                    'chunk': current_chunk,
                    'split_threshold': split_threshold,
                    'token_num': current_chunk_token_num
                })
                # 重置当前chunk
                current_chunk = f'{f"------page{i}------" if add_page else ""}\n{f"{j}|" if add_row_number else ""}{line}'
                current_chunk_token_num = current_line_token_num
                current_page_number_list = [i]
            else:
                # 向当前chunk添加内容
                current_chunk += f'\n{f"{j}|" if add_row_number else ""}{line}'
                current_chunk_token_num += current_line_token_num
                if i not in current_page_number_list:
                    current_page_number_list.append(i)

    chunks.append({'book': book_path,
                   'page_number_list': current_page_number_list,
                   'chunk': current_chunk,
                   'split_threshold': split_threshold,
                   'token_num': current_chunk_token_num})


print(len(chunks))
write_jsonl(args.save_path, chunks)
