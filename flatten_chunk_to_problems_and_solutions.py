from utils import load_jsonl, write_jsonl
from multiprocessing import Pool
import argparse
import openai
import time
import random


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', required=False, default='/gemini/space/ludakuan/data/book_chunks/all_problems_and_solutions_extracted_by_o4-mini.jsonl', help='Path to input problems_and_solutions jsonl file')
parser.add_argument('--output_path', required=False, default='/gemini/space/ludakuan/data/extracted_problems/all_problems_and_solutions_extracted_by_o4-mini.jsonl', help='Path to save filtered output jsonl file')
args = parser.parse_args()

chunks = load_jsonl(args.input_path)
total_chunk_num = len(chunks)
print('total chunk number:', total_chunk_num)
chunks = [chunk for chunk in chunks if 'problems_and_solutions' in chunk and len(chunk['problems_and_solutions']) > 0]
have_problems_and_solutions_num = len(chunks)
print('have problems and solutions number:', have_problems_and_solutions_num)


# 1.处理\\的格式问题
for i in range(len(chunks)):
    for j in range(len(chunks[i]['problems_and_solutions'])):
        try:
            if 'problem' in chunks[i]['problems_and_solutions'][j]:
                chunks[i]['problems_and_solutions'][j]['problem'] = chunks[i]['problems_and_solutions'][j]['problem'].replace('\\\\', '\\')
            if 'solution' in chunks[i]['problems_and_solutions'][j]:
                chunks[i]['problems_and_solutions'][j]['solution'] = chunks[i]['problems_and_solutions'][j]['solution'].replace('\\\\', '\\')
        except Exception as e:
            print(e)
            continue

# 2.展开problems and solutions
problems_and_solutions = []
for chunk in chunks:
    for p_or_s in chunk['problems_and_solutions']:
        if not isinstance(p_or_s, dict):
            continue
        if 'problem number' in p_or_s:
            p_or_s['problem number'] = str(p_or_s['problem number'])
        if 'solution number' in p_or_s:
            p_or_s['solution number'] = str(p_or_s['solution number'])
        if ('problem' in p_or_s and 'problem number' in p_or_s and isinstance(p_or_s['problem'], str)) or ('solution' in p_or_s and 'solution number' in p_or_s and isinstance(p_or_s['solution'], str)):
            try:
                p_or_s['book'] = chunk['book']
                p_or_s['chunk_number'] = chunk['chunk_number']
                p_or_s["page_number_list"] = chunk["page_number_list"]
                problems_and_solutions.append(p_or_s)
            except Exception as e:
                print(e)
                continue

print('lenth of p and s:', len(problems_and_solutions))
# problem number
problem_num = len([p_or_s for p_or_s in problems_and_solutions if 'problem' in p_or_s])
print('problem number:', problem_num)
# solution number
solution_num = len([p_or_s for p_or_s in problems_and_solutions if 'solution' in p_or_s])
print('solution number:', solution_num)
# save all problems and solutions
write_jsonl(args.output_path, problems_and_solutions)
