from utils import *
import random
from openai import AsyncOpenAI
import argparse
import time
from tqdm.asyncio import tqdm as async_tqdm
import asyncio
import re


async def judge_problems_and_solutions_unavailable(problem, solution, book, client, model_name, semaphore):
    prompt = prompt = f'''1. Task Overview
I have extracted problems and their possible solutions from textbooks using ocr, extraction and retrieval algorithms for llm training. 
Please help me determine:
对于problem和solution的判断，遵循不同的规则：
For the problem:
    - contains references to invisible content (such as formulas, diagrams, other problems and answers, etc.) 
    - **AND these missing invisible contents affect the LLM's understanding of the problem**
    - those that do not affect understanding can be ignored
    - 对于需要记忆的公式、定理、常数、定义和概念，默认LLM含有这些知识，因此这些内容不会影响理解。
    - If it contains and affects the understanding of the problem and answer by LLM, return True, otherwise return False.
For the solution:
    - solution最终只被用于在强化学习奖励模型中判断对错，而不是直接用于监督学习。因此对于solution的判断要相对宽松一点。
    - 首先我们要判断题型，然后根据题型判断是否可用。
    - 对于选择题、填空题和计算题，**只要包含有可以用于判断对错的最终答案**，就可以判断为可用。
    - 对于证明题，只要包含有**可以判断证明对错的关键证明步骤和结论**，就可以判断为可用。
    - 对于问答题，只要包含有可以判断答案对错的关键要点，就可以判断为可用。
    - 对于多个小问的题目，如果有任何一个小问的答案缺失或不可用，则整个解答不可用。
    - 如果回答中包含有多个问题的答案，且无法准确根据题号找到当前问题的答案，则整个解答不可用。
2. Examples
    Example 1:
        Source: ../zh_xiti_books_science/无机化学习题解答  第3版(张丽荣，于杰辉，王莉，宋天佑编).jsonl
        Problem:1-2 已知 $1\ \mathrm{{dm}}^{{3}}$ 某气体在标准状况下质量为 $2.86\ \mathrm{{g}}$, 试计算该气体的平均相对分子质量, 并计算其在 $17^{{\circ}}\mathrm{{C}}$ 和 $207\mathrm{{kPa}}$ 时的密度。
        Solution:先计算此气体的物质的量 $n_{{0}}$.  由理想气体状态方程 $pV=nRT$ 得  
        $$n=\\frac{{pV}}{{RT}}$$  
        依题意 $V=1\ \mathrm{{dm}}^{{3}}=1\\times10^{{-3}}\ \mathrm{{m}}^{{3}},\ p=101.3\ \mathrm{{kPa}},\ T=273\ \mathrm{{K}}$, 代入求出 $n$:  
        $$n=\\frac{{101.3\ \mathrm{{kPa}}\\times1\\times10^{{-3}}\ \mathrm{{m}}^{{3}}}}{{8.314\ \mathrm{{J}}\cdot\mathrm{{mol}}^{{-1}}\cdot\mathrm{{K}}^{{-1}}\\times273\ \mathrm{{K}}}}=0.04463\ \mathrm{{mol}}$$  
        由题设知 0.04463 mol 气体的质量为 2.86 g, 故该气体的摩尔质量为  
        $$M=64.1\\ \mathrm{{g}}\cdot\mathrm{{mol}}^{{-1}}$$  
        即平均相对分子质量为 64.1。  
        设 $17^{{\circ}}\mathrm{{C}}$ 和 $207\mathrm{{kPa}}$ 时气体的体积为 $V$, 由理想气体状态方程知:$$V=\\frac{{nRT}}{{p}}$$  
        代入题设条件  
        $$\\begin{{aligned}}V&=\\frac{{0.04463\ \mathrm{{mol}}\times8.314\ \mathrm{{J}}\cdot\mathrm{{mol}}^{{-1}}\cdot\mathrm{{K}}^{{-1}}\\times(273+17)\ \mathrm{{K}}}}{{207000\ \mathrm{{Pa}}}}\\&=5.20\\times10^{{-4}}\ \mathrm{{m}}^{{3}}\end{{aligned}}$$  
        即气体的体积为 $0.520\ \mathrm{{dm}}^{{3}}$.  
        设气体的密度为 $\\rho$, 则  
        $$\\rho=\\frac{{m}}{{V}}=\\frac{{2.86\ \mathrm{{g}}}}{{0.520\ \mathrm{{dm}}^{{3}}}}=5.50\\ \mathrm{{g}}\cdot\mathrm{{dm}}^{{-3}}$$
        Judgment:1. 内容完备性：题干已给出所有必要数值（1 dm³、2.86 g、17 ℃、207 kPa），标准状况（STP）为常识性参数，无需外部图表。
        2. 无外部引用：文本中未出现“如图所示”、“见表1-1”或“参考前一题”等指向缺失视听内容的描述。
        结论：\\boxed{{False}}。
    Example 2:
        Source:../zh_xiti_books_science/普通物理学问题与习题集(И.В.萨韦利耶夫; 贺锡纯).jsonl
        Problem:有两个斜面，它们跟半径为 $R$ 的同一圆周的两条弦重合(图 1.12)。一小物体由静止开始分别从这两个斜面无摩擦地滑落。试问在哪一个斜面上滑落的时间较长些？
        Solution:对两个斜面来说滑落的时间相同。
        Judgment:1. 显式引用缺失：题干中明确提到了“(图 1.12)”，但 OCR 提取内容中并未包含该图像或其详细描述。
        2. 几何条件不确定：这个题目看起来是在考察物理学中的“等时圆”原理（物体沿圆周弦滑下时间相等），但是这通常要求弦具有共同的顶点（如均从最高点出发或均到达最低点），题目文字中并没有明确提到。
        3. 逻辑验证受阻：仅凭文字描述“两条弦”而没有图形参考，无法确定这两条弦的具体空间分布和起始位置。对于 LLM 而言，在缺乏图形的情况下，无法严谨判断“时间相同”这一结论的适用条件是否成立。
        4. 关键信息遗漏：图像是本题几何模型的核心组成部分，缺失该图像直接影响了对物理场景的复原，且无法通过文字描述和已知条件推导出。
        结论：\\boxed{{True}}。
    Example 3:
        Source:../math_problems_and_questions_book_609/2500 Solved Problems in Differential Equations(Richard Bronson).jsonl
        Problem:Solve \[ y'' - 2y' + y = e^x/x^3 \].
        Solution:The complementary solution \(y_c\), and \(y_1,y_2\) of the previous problem remain valid, and we assume\n\n\[ y_p = t_1 e^x + t_2 x e^x, \]\n\nwith \(\phi(x)= e^x/x^3\). From Problem 10.3:\n\n\[ t_1' e^x + t_2' x e^x = 0, \]\n\[ t_1' e^x + t_2'(e^x + x e^x) = \\frac{{e^x}}{{x^3}}. \]\n\nSolving yields \( t_1' = -1/x^2 \), \( t_2' = 1/x^3 \), and so\n\n\[\n v_1 = \int -1/x^2\,dx = 1/x, \quad v_2 = \int 1/x^3\,dx = -frac1{{2x^2}}.\n\]\n\nThus\n\n\[ y_p = \\frac1x e^x + \Bigl(-frac1{{2x^2}}\Bigr) x e^x = frac1{{2x}} e^x, \]\n\nand\n\n\[ y = y_c + y_p = c_1 e^x + c_2 x e^x + frac1{{2x}} e^x. \]
        Judgment:首先，对于问题，是一个完整的微分方程问题，需要求解。因此，问题本身是可用的。
        其次，对于solution，虽然包含了对之前别的微分方程问题的引用，但是按照solution的规则，它给出了可以用于判断对错的最终答案，因此solution也是可用的。
        结论：\\boxed{{False}}。
3. Input
Source:

---

{book}

---

Problem:

---

{problem}

---

Solution:

---

{solution}

---

3. Output Format
Let's think step by step. Show your reasoning process and provide your final judgment in the following format
\\boxed{{True}} or \\boxed{{False}}
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
                
                # print(prompt, response, '=' * 10, sep='\n---\n')
                
                # 使用正则表达式提取最后一个 \boxed{} 中的内容
                matches = re.findall(r'\\boxed\{([^}]+)\}', response)
                if matches:
                    # 获取最后一个匹配
                    result = matches[-1].strip()
                    # 判断是否为 True
                    if result.lower() == 'true':
                        return True
                    else:
                        return False
                else:
                    # 如果没有找到 \boxed{} 格式，返回 False
                    print(f'Warning: No \\boxed{{}} found in response: {response[:100]}...')
                    return False
            except Exception as e:
                print(f'Error: {e}')
                retries += 1
                if retries > max_retries:
                    return False
                await asyncio.sleep(30)  # 等待重试


async def data_process(data, client, model_name, semaphore):
    # 处理一些遗留格式问题
    if 'problem number' in data:
        if data['problem'].startswith(data['problem number'] + '. '):
            data['problem'] = data['problem'][len(data['problem number'] + '. '):]
    if 'solution number' in data['matched_solution']:
        if data['matched_solution']['solution'].startswith(data['matched_solution']['solution number'] + '. '):
            data['matched_solution']['solution'] = data['matched_solution']['solution'][len(data['matched_solution']['solution number'] + '. '):]
    data['matched_solution'] = data['matched_solution']['solution']

    """异步处理单个数据项"""
    is_unavailable = await judge_problems_and_solutions_unavailable(
        data['problem'], 
        data['matched_solution'],
        data['book'],
        client,
        model_name,
        semaphore
    )
    data['is_unavailable'] = is_unavailable
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
    parser.add_argument('--input_file', type=str, default='/gemini/space/guarded_files/ludakuan/data/extracted_problems/all_problems_had_matched_solutions.jsonl')
    # 最大并发数（控制同时发起的API请求数量）
    parser.add_argument('--max_concurrent', type=int, default=128, help='最大并发请求数，建议根据vllm服务器性能设置，默认128')
    # 总分片数与当前分片
    parser.add_argument('--total_slice', type=int, default=100)
    parser.add_argument('--current_slice', type=int, default=0)
    # output file
    parser.add_argument('--output_path', type=str, default='/gemini/space/guarded_files/ludakuan/data/extracted_problems/')
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
    
    write_jsonl(os.path.join(args.output_path, f"all_problems_with_matched_solutions_filtered_slice_{args.current_slice}_of_{args.total_slice}.jsonl"), results)


if __name__ == '__main__':
    # 运行异步主函数
    asyncio.run(main())
