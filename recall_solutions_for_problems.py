import json
import random
import argparse

import tiktoken
from tqdm import tqdm

from utils import load_jsonl, write_jsonl
from multiprocessing import Pool
from gpt4_request import request_one_turn, safe_request_one_turn

from sentence_transformers import SentenceTransformer
import numpy as np
import torch


# Add argument parser
def parse_args():
    parser = argparse.ArgumentParser(description='Process problems and solutions with embeddings')
    parser.add_argument('--input_file', type=str, required=True,
                      help='Input JSONL file containing problems and solutions')
    parser.add_argument('--output_file', type=str, required=True,
                      help='Output JSONL file for problems with recalled solutions')
    return parser.parse_args()

# Add args at the beginning of main logic
args = parse_args()
problems_and_solutions = load_jsonl(args.input_file)
# calculate total size, problems and solutions count
problems_count = 0
solutions_count = 0
for p_or_s in problems_and_solutions:
    if 'problem' in p_or_s:
        problems_count += 1
    if 'solution' in p_or_s:
        solutions_count += 1
print(f"total size: {len(problems_and_solutions)}, problems count: {problems_count}, solutions count: {solutions_count}")
# data examples:
# {"problem number": "1.48", "problem": "A genetic variant is indicated as follows: NM_006735.4(HOXA2):c.394T>A (p.Ser132Thr) Which part of this phrasing refers to the specific genetic change in the reference sequence of the coding DNA? A. NM_006735.4 B. HOXA2 C. c.394T>A D. p.Ser132Thr", "book": "./docs/MEDICAL GENETICS AND GENOMICS.jsonl", "chunk_number": 2, "page_number_list": [16, 17, 18, 19], "is_bad": false}
# {"solution number": "1.48", "solution": "C", "book": "./docs/MEDICAL GENETICS AND GENOMICS.jsonl", "chunk_number": 2, "page_number_list": [16, 17, 18, 19], "is_bad": false}


# type unify to solve bug
print("type unify to solve bug")
for p_or_s in problems_and_solutions:
    if 'problem number' in p_or_s:
        p_or_s['problem number'] = str(p_or_s['problem number'])
    if 'problem' in p_or_s:
        p_or_s['problem'] = str(p_or_s['problem'])
    if 'solution number' in p_or_s:
        p_or_s['solution number'] = str(p_or_s['solution number'])
    if 'solution' in p_or_s:
        p_or_s['solution'] = str(p_or_s['solution'])
    if 'book' in p_or_s:
        p_or_s['book'] = str(p_or_s['book'])

# 拆开那些problem和solution在一个字典里的
print("split the problem and solution in the same dictionary")
problems_and_solutions_dedup = []
for p_or_s in problems_and_solutions:
    if 'problem' in p_or_s:
        problems_and_solutions_dedup.append(
            {
                'problem number': p_or_s['problem number'],
                'problem': p_or_s['problem'],
                'book': p_or_s['book'],
                'chunk_number': p_or_s['chunk_number'],
                'page_number_list': p_or_s['page_number_list'],
            }
        )
    if 'solution' in p_or_s:
        try:
            problems_and_solutions_dedup.append(
                {
                    'solution number': p_or_s['solution number'] if 'solution number' in p_or_s else p_or_s['problem number'],
                    'solution': p_or_s['solution'],
                    'book': p_or_s['book'],
                    'chunk_number': p_or_s['chunk_number'],
                    'page_number_list': p_or_s['page_number_list'],
                }
            )
        except Exception as e:
            print(e)
            print(json.dumps(p_or_s, indent=4))
            raise e

problems_and_solutions = problems_and_solutions_dedup


# 将problem number和solution number放在problem和solution前面
print("put problem number and solution number in front of problem and solution")
for p_or_s in problems_and_solutions:
    if 'problem number' in p_or_s:
        p_or_s['problem'] = p_or_s['problem number'] + '. ' + p_or_s['problem']
    if 'solution number' in p_or_s:
        p_or_s['solution'] = p_or_s['solution number'] + '. ' + p_or_s['solution']

# get embedding for problems and solutions
# 初始化模型
print("initialize model")
model = SentenceTransformer("dunzhang/stella_en_400M_v5", 
                        trust_remote_code=True).cuda()

# batch size
BATCH_SIZE = 128

def batch_encode(texts, indices, prompt_name=None):
    """批量编码文本"""
    embeddings = []
    
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Encoding"):
        batch_texts = texts[i:i + BATCH_SIZE]
        if prompt_name:
            batch_embeddings = model.encode(batch_texts, prompt_name=prompt_name)
        else:
            batch_embeddings = model.encode(batch_texts)
        embeddings += [emb for emb in batch_embeddings]
        
        # 可选：定期清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 将embedding存储回原始数据
    for i, idx in enumerate(indices):
        problems_and_solutions[idx]['embedding'] = embeddings[i]

# 准备问题和答案文本
problems = []
solutions = []
problem_indices = []
solution_indices = []

for i, item in enumerate(problems_and_solutions):
    if 'problem' in item:
        problems.append(item['problem'])
        problem_indices.append(i)
    if 'solution' in item:
        solutions.append(item['solution'])
        solution_indices.append(i)

# 分批处理问题和答案
print("Processing problems...")
batch_encode(problems, problem_indices, prompt_name="s2p_query")
print("Processing solutions...")
batch_encode(solutions, solution_indices)


K = 4
# 根据题号和相似度召回题解
for i in tqdm(range(len(problems_and_solutions))):
    if 'problem' in problems_and_solutions[i] and 'embedding' in problems_and_solutions[i]:
        recalled_solutions = []
        for j in range(i, len(problems_and_solutions)):
            if problems_and_solutions[j]['book'] != problems_and_solutions[i]['book']:
                break
            if 'solution' in problems_and_solutions[j] and 'solution number' in problems_and_solutions[j]:
                if problems_and_solutions[j]['solution number'] == problems_and_solutions[i]['problem number']:
                    # 只保留编号和solution number
                    solution_copy = {
                        'solution number': problems_and_solutions[j]['solution number'],
                        'solution': problems_and_solutions[j]['solution'],
                    }
                    recalled_solutions.append((solution_copy, 1))
                    continue
            if 'solution' in problems_and_solutions[j] and 'embedding' in problems_and_solutions[j]:
                similarities = model.similarity(problems_and_solutions[i]['embedding'], problems_and_solutions[j]['embedding'])
                solution_copy = {
                    'solution number': problems_and_solutions[j]['solution number'],
                    'solution': problems_and_solutions[j]['solution'],
                }
                recalled_solutions.append((solution_copy, similarities))
        
        if recalled_solutions:
            recalled_solutions.sort(key=lambda x: x[1], reverse=True)
            problems_and_solutions[i]['recalled_solutions'] = [item[0] for item in recalled_solutions[:K]]
        problems_and_solutions[i].pop('embedding', None)

# 只保留问题和召回的题解
problems_with_solutions = [item for item in problems_and_solutions if 'problem' in item]

write_jsonl(args.output_file, problems_with_solutions)
