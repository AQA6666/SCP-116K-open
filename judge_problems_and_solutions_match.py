from utils import *
from gpt4_request import safe_request_one_turn
from multiprocessing import Pool
import random
import argparse


def judge_problems_and_solutions_match(problem, solution):
    prompt = f'''1. Task Overview
I have extracted problem-solution pairs from textbooks using extraction and matching algorithms. Please help me determine if the following problem and solution constitute a 'valid' problem-solution pair.

2. Input
Problem:
---
{problem}
---
Solution:
---
{solution}
---

3. Evaluation Process
    a. First, verify that the problem is indeed a 'problem' and the solution is a 'solution', not other content
    b. Then, confirm that the problem and solution match - the solution specifically addresses this problem, not some other problem
    c. Finally, check if the solution is correct and complete. The solution can contain only the final answer without the solving process, but must have a final answer.
       'Complete' means the solution does not reference other invisible information, such as formulas, diagrams, and answers from other problems, and can be independently understood and verified (if the missing information does not affect understanding and verification, it can be ignored).
       'Correct' means the final result of the solution is correct. **You must verify the correctness of the solution through rigorous reasoning.** If it cannot be verified, return False.

4. Output Format
Let's think step by step. Show your reasoning process and provide your final judgment in the following format, where 'True' means the problem and solution constitute a 'valid' problem-solution pair:
[Begin]True/False[End]
'''
    try:
        response = safe_request_one_turn(prompt, model='gpt-4o', error_response='[Begin]False[End]')
        print(prompt, response, sep='\n---\n')
        print('=' * 10)
        if '[Begin]True[End]' in response:
            return True
        else:
            return False
    except:
        return False


def data_process(data):
    data['problem'] = data['problem'][data['problem'].find('. ') + 2:]
    if 'recalled_solutions' in data:
        for sol in data['recalled_solutions']:
            sol['solution'] = sol['solution'][sol['solution'].find('. ') + 2:]
            if judge_problems_and_solutions_match(data['problem'], sol['solution']):
                data['matched_solution'] = sol
                break
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Match problems with solutions from JSONL files')
    parser.add_argument('--input', type=str, required=True,
                      help='Input JSONL file path containing problems and recalled solutions')
    parser.add_argument('--output', type=str, required=True,
                      help='Output JSONL file path for problems with matched solutions')
    args = parser.parse_args()

    data = load_jsonl(args.input)
    # data = random.sample(data, 100)
    with Pool() as pool:
        data = pool.map(data_process, data)
    matched_data = [d for d in data if 'matched_solution' in d]
    print(f'{len(matched_data)}/{len(data)} problems have matched solutions')
    write_jsonl(args.output, data)
