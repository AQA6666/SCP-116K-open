import json
import tiktoken
import argparse
from tqdm import tqdm
from utils import load_jsonl, write_jsonl
from multiprocessing import Pool
from gpt4_request import request_one_turn, safe_request_one_turn

# Add argument parser
parser = argparse.ArgumentParser(description='Filter problems and solutions from JSONL file')
parser.add_argument('--input_file', type=str, required=True,
                    help='Path to input JSONL file containing problems and solutions')
parser.add_argument('--output_file', type=str, required=True,
                    help='Path to output filtered JSONL file')
args = parser.parse_args()

# Replace hardcoded paths with command line arguments
chunks = load_jsonl(args.input_file)
print(len(chunks))
chunks = [chunk for chunk in chunks if 'problems_and_solutions' in chunk]
print(len(chunks))
# chunks = chunks[:100]

# 1.处理\\的格式问题
for i in range(len(chunks)):
    for j in range(len(chunks[i]['problems_and_solutions'])):
        try:
            if 'problem' in chunks[i]['problems_and_solutions'][j]:
                chunks[i]['problems_and_solutions'][j]['problem'] = chunks[i]['problems_and_solutions'][j]['problem'].replace('\\\\', '\\')
            elif 'solution' in chunks[i]['problems_and_solutions'][j]:
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


# 3.过滤所有有问题的problem and solution
def filter_bad_problem(problem):
    instruction = f'''Problem:
---
{problem['problem']}
---
This is a problem extracted from textbook content. Please follow the conditional branching process below to help determine whether this problem's quality is good or poor.
If the input content is not a problem, but rather content like solutions, answers, or formulas:
    then quality = poor, end
Otherwise if the input content is a problem:
    attempt to solve the problem:
        if it cannot be solved due to missing essential information (such as missing formulas, images, problem statements, or other questions), then quality = poor, end
        otherwise if the problem can be solved, then quality = good, end
Let's think step by step. First, describe your thought process, then provide the answer in the following format:
[Begin Tag]Good/Poor[End Tag]
'''
    response = safe_request_one_turn(prompt=instruction, error_response='[Begin Tag]Poor[End Tag]')
    # print(instruction, response)
    # print('---')
    try:
        response = response.split('[Begin Tag]')[-1].split('[End Tag]')[0]
    except Exception as e:
        response = 'Poor'
    if 'Poor' in response:
        problem['is_bad'] = True
    else:
        problem['is_bad'] = False
    return problem


def filter_bad_solution(solution):
    instruction = f'''Solution:
    ---
    {solution['solution']}
    ---
I extracted some solutions from textbooks using OCR and extraction algorithms. Please help me determine if the given solutions have any of the following quality issues:
1. Incorrectly extracting problems as solutions, that is, first, we need to determine whether the provided content is a solution or a problem.
2. There are numerous OCR recognition errors making it incomprehensible (a small number of typos can be ignored).
3. Necessary images, formulas or other necessary information are missing, leading to incomprehensibility (unnecessary images and formulas can be ignored).
If any of the above situations exist, then the quality is considered problematic.
Let's think step by step. First, describe your thought process, then provide the answer in the following format, where Yes indicates that there is a quality issue:
[Begin Tag]Yes/No[End Tag]
    '''
    response = safe_request_one_turn(prompt=instruction, error_response='[Begin Tag]Yes[End Tag]')
    # print(instruction, response)
    # print('---')
    try:
        response = response.split('[Begin Tag]')[-1].split('[End Tag]')[0]
    except IndexError:
        response = 'Yes'
    if 'Yes' in response:
        solution['is_bad'] = True
    else:
        solution['is_bad'] = False
    return solution


def problem_and_solution_filter(p_or_s):
    if 'problem' in p_or_s:
        p_or_s = filter_bad_problem(p_or_s)
    elif 'solution' in p_or_s:
        p_or_s = filter_bad_solution(p_or_s)
    else:
        p_or_s = {'is_bad': True}
    return p_or_s


with Pool(processes=64) as pool:
    problems_and_solutions = pool.map(problem_and_solution_filter, problems_and_solutions)
print('initial problem number:', len([p_or_s for p_or_s in problems_and_solutions if 'problem' in p_or_s]))
problems_and_solutions = [p_or_s for p_or_s in problems_and_solutions if not p_or_s['is_bad']]
print('after filter problem number:', len([p_or_s for p_or_s in problems_and_solutions if 'problem' in p_or_s]))

# Replace hardcoded output path
write_jsonl(args.output_file, problems_and_solutions)
