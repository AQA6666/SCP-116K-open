import json
import random
import argparse
from tqdm import tqdm
from utils import load_jsonl, write_jsonl
from sentence_transformers import SentenceTransformer
import torch


# 创建命令行参数解析器
parser = argparse.ArgumentParser(description='处理问题和解答的相似度计算与召回')
parser.add_argument('--input_path', type=str, 
                    default='/gemini/space/ludakuan/data/extracted_problems/all_problems_and_solutions_extracted_by_o4-mini.jsonl',
                    help='输入数据路径')
parser.add_argument('--output_path', type=str, 
                    default='/gemini/space/ludakuan/data/extracted_problems/all_problems_with_recalled_solutions.jsonl',
                    help='输出数据保存路径')
parser.add_argument('--model_path', type=str, 
                    default='Qwen/Qwen3-Embedding-4B',
                    help='相似度计算模型路径')
parser.add_argument('--model_cache_dir', type=str, 
                    default='/gemini/space/ludakuan/model/Qwen3-Embedding-4B',
                    help='模型缓存目录')
parser.add_argument('--batch_size', type=int, default=128, help='批处理大小')
parser.add_argument('--top_k', type=int, default=8, help='召回解答数量')
parser.add_argument('--sample_limit', type=int, default=10000, help='处理样本数量限制，0表示不限制')

# 解析命令行参数
args = parser.parse_args()

# 加载数据
problems_and_solutions = load_jsonl(args.input_path)
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
# {"problem number": "1.48", "problem": "A genetic variant is indicated as follows: NM_006735.4(HOXA2):c.394T>A (p.Ser132Thr) Which part of this phrasing refers to the specific genetic change in the reference sequence of the coding DNA? A. NM_006735.4 B. HOXA2 C. c.394T>A D. p.Ser132Thr", "book": "./Books/Biology/MEDICAL GENETICS AND GENOMICS  questions for board review. (BENJAMIN D. SOLOMON) (Z-Library).jsonl", "chunk_number": 2, "page_number_list": [16, 17, 18, 19], "is_bad": false}
# {"solution number": "1.48", "solution": "C", "book": "./Books/Biology/MEDICAL GENETICS AND GENOMICS  questions for board review. (BENJAMIN D. SOLOMON) (Z-Library).jsonl", "chunk_number": 2, "page_number_list": [16, 17, 18, 19], "is_bad": false}

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

# 限制处理样本数量
if args.sample_limit > 0:
    problems_and_solutions = problems_and_solutions_dedup[:args.sample_limit]
else:
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
model = SentenceTransformer(args.model_path, 
                        trust_remote_code=True, 
                        cache_folder=args.model_cache_dir).cuda()

# batch size
BATCH_SIZE = args.batch_size

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
batch_encode(problems, problem_indices, prompt_name="query")
print("Processing solutions...")
batch_encode(solutions, solution_indices)


K = args.top_k
# 根据题号和相似度召回题解
for i in tqdm(range(len(problems_and_solutions))):
    if 'problem' in problems_and_solutions[i] and 'embedding' in problems_and_solutions[i]:
        recalled_solutions = []
        for j in range(i + 1, len(problems_and_solutions)):
            if problems_and_solutions[j]['book'] != problems_and_solutions[i]['book']:
                break
            if 'solution' in problems_and_solutions[j] and 'solution number' in problems_and_solutions[j] and 'embedding' in problems_and_solutions[j]:
                similarity = model.similarity(problems_and_solutions[i]['embedding'], problems_and_solutions[j]['embedding'])
                if problems_and_solutions[j]['solution number'] == problems_and_solutions[i]['problem number']:
                    similarity += 0.1
                # 只保留编号和solution number
                solution_copy = {
                    'solution number': problems_and_solutions[j]['solution number'],
                    'solution': problems_and_solutions[j]['solution'],
                }
                recalled_solutions.append((solution_copy, similarity))
                # 如果题解与题目相邻则短路退出遍历
                if problems_and_solutions[j]['solution number'] == problems_and_solutions[i]['problem number'] and j == i + 1:
                    break
        
        if recalled_solutions:
            recalled_solutions.sort(key=lambda x: x[1], reverse=True)
            problems_and_solutions[i]['recalled_solutions'] = [item[0] for item in recalled_solutions[:K]]
        problems_and_solutions[i].pop('embedding', None)

# 只保留问题和召回的题解
problems_with_solutions = [item for item in problems_and_solutions if 'problem' in item]

write_jsonl(args.output_path, problems_with_solutions)
