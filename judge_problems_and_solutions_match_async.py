from utils import *
import random
from openai import AsyncOpenAI
import argparse
import time
from tqdm.asyncio import tqdm as async_tqdm
import asyncio


async def judge_problems_and_solutions_match(problem, solution, client, model_name, semaphore):
    """异步判断问题和解决方案是否匹配"""
    prompt = f'''1. Task Overview

I have extracted problems and their possible solutions from textbooks using extraction and retrieval algorithms. Please help me determine if the following problem and solution constitute a 'valid' problem-solution pair.

2. Input
Problem:

---

{problem}

---

Solution:

---

{solution}

---

3. Reference for common judgment methods
    a. Verify that the problem number and solution number match if there are numbers in both the problem and solution. Sometimes the both two numbers are not strictly same:
        Example 1: The problem number is '1.1' and the solution number is '9.1'. It is likely that all solutions are placed in chapters after the problem chapters. So it is likely that the solution is the solution to the problem '1.1'.
        Example 2: The problem number is '1.1' and the solution number is '1'. It is likely that in the extraction process, the solution number is partially missed. So it is likely that the solution is the solution to the problem '1.1'.
        Example 3: The problem number is '1.1' and the solution number is '1.2' or '3.4'. It is likely that the recall process occurred error. So it is unlikely that the solution is the solution to the problem '1.1'.
    b. Confirm that the content of the problem and solution match - the solution specifically addresses this problem, not some other problem:
        (1) If the problem is a multiple-choice question and the solution only has numbers or formulas, then there is a high probability that it does not match. 
        (2) If the question is about speed and the solution is about quality, then there is a high probability of a mismatch. 
        (3) If the problem is a proof problem and the solution is a calculation process, then there is a high probability that it is also a mismatch
    c. For calculation problems, you can substitute the answer back into the problem to check whether it is correct. For proof problems, as long as there are key proof steps and conclusions, it can be judged as a match. 
    d. Both problems and solutions are recognized using OCR, so there may be some minor errors in details, such as incorrect recognition of symbols and numbers. Even if most of the problem-solving process is correct, if there are fatal symbol and numerical errors in the final answer, it should still be judged as a mismatch.
    e. Both problems and solutions may contain references to invisible content, such as formulas, figures, tables, etc. If the content of the invisible references makes the problem and solution incomprehensible or impossible to determine whether they match, it is also judged as a mismatch.
        Example 4: 
            Problem: Use the formula from problem 4.18 to calculate the values of a, b, c in problem 4.19
            Solution: $a=1000$, $b=2000$, $c=3000$.
            Judgment: Although the problem asks for the values of a, b, c and the solution provides the values of a, b, c, since the problem only contains references to invisible content, we cannot understand its content or substitute it for checking, therefore it is judged as a mismatch
    f. If the problem has multiple sub-problems and the solution is missing answers because of naturally missing or reference to invisible content for more than 1/3 of the sub-problems, it is also judged as a mismatch.
4. Output Format
Let's think step by step. Show your reasoning process and provide your final judgment in the following format, where 'True' means the problem and solution constitute a 'valid' problem-solution pair:
[Begin]True/False[End]
'''
    max_retries = 2
    retries = 0
    
    # 使用信号量控制并发
    async with semaphore:
        while retries <= max_retries:
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.6,
                )
                
                response = response.choices[0].message.content
                
                # print(prompt, response, sep='\n---\n')
                # print('=' * 10)
                if '[Begin]True[End]' in response:
                    return True
                else:
                    return False
            except Exception as e:
                print(f'Error: {e}')
                retries += 1
                if retries > max_retries:
                    return False
                await asyncio.sleep(30)  # 等待重试


async def data_process(data, client, model_name, semaphore):
    """异步处理单个数据项"""
    if 'recalled_solutions' in data:
        for sol in data['recalled_solutions']:
            is_match = await judge_problems_and_solutions_match(
                data['problem'], 
                sol['solution'],
                client,
                model_name,
                semaphore
            )
            if is_match:
                data['matched_solution'] = sol
                break
        data.pop('recalled_solutions')
    return data


async def main():
    """主异步函数"""
    parser = argparse.ArgumentParser()
    # model name
    parser.add_argument('--model_name', type=str, default='Qwen3-Next-80B-A3B-Instruct')
    # api key
    parser.add_argument('--api_key', type=str, default='dummy')
    # base url
    parser.add_argument('--base_url', type=str, default="http://localhost:8998/v1")
    # input file
    parser.add_argument('--input_file', type=str, default='/gemini/space/ludakuan/data/extracted_problems/all_problems_with_recalled_solutions_shuffled.jsonl')
    # 最大并发数（控制同时发起的API请求数量）
    parser.add_argument('--max_concurrent', type=int, default=128, help='最大并发请求数，建议根据vllm服务器性能设置，默认128')
    # 总分片数与当前分片
    parser.add_argument('--total_slice', type=int, default=100)
    parser.add_argument('--current_slice', type=int, default=0)
    # output file
    parser.add_argument('--output_path', type=str, default='/gemini/space/ludakuan/data/extracted_problems/')
    args = parser.parse_args()

    # 遍历打印参数信息
    for key, value in args.__dict__.items():
        print(f"{key}: {value}")
    
    # 创建异步OpenAI客户端
    client = AsyncOpenAI(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout=600,
    )
    
    # 创建信号量来控制并发数
    semaphore = asyncio.Semaphore(args.max_concurrent)
    
    data = load_jsonl(args.input_file)
    data = data[len(data) * args.current_slice // args.total_slice:
                          len(data) * (args.current_slice + 1) // args.total_slice]
    # 打印总分片数，当前分片数，当前分片数据长度
    print(f"总分片数: {args.total_slice}")
    print(f"当前分片数: {args.current_slice}")
    print(f"当前分片数据长度: {len(data)}")
    print(f"最大并发数: {args.max_concurrent}")
    
    begin_time = time.time()
    
    # 创建所有异步任务
    tasks = [data_process(item, client, args.model_name, semaphore) for item in data]
    
    # 使用 tqdm 显示进度并执行所有任务
    results = []
    for coro in async_tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="处理进度"):
        result = await coro
        results.append(result)
    
    end_time = time.time()
    print(f"处理时间: {end_time - begin_time}秒")
    print(f"平均每个数据项处理时间: {(end_time - begin_time) / len(data):.2f}秒")
    
    write_jsonl(os.path.join(args.output_path, f"all_problems_with_matched_solutions_slice_{args.current_slice}_of_{args.total_slice}.jsonl"), results)


if __name__ == '__main__':
    # 运行异步主函数
    asyncio.run(main())
