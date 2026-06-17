import random
import time
from utils import load_jsonl, write_jsonl, find_files
from gpt4_request import request_one_turn_with_one_image
import fitz  # PyMuPDF
import base64
from io import BytesIO
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm
from clean_book_page_md_tag import clean_text
import argparse
from openai import OpenAI
import os


def transfer_pdf_to_base64_images(pdf_path, max_size_mb=18):
    max_size_bytes = max_size_mb * 1024 * 1024  # 转为字节
    base64_images = []

    # 打开 PDF 文件
    pdf_document = fitz.open(pdf_path)

    for page_number in range(len(pdf_document)):
        # 获取页面
        page = pdf_document.load_page(page_number)
        width, height = page.rect.width, page.rect.height

        # 计算动态缩放因子
        max_scale = (max_size_bytes / (4 / 3 * width * height * 3)) ** 0.5

        # 设置缩放矩阵
        matrix = fitz.Matrix(max_scale, max_scale)

        # 渲染页面为像素图
        pix = page.get_pixmap(matrix=matrix)

        # 将像素图转为PIL图像
        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # 将图片保存到内存
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        buffer.seek(0)

        # 转为 Base64 编码
        base64_image = base64.b64encode(buffer.read()).decode("utf-8")
        base64_images.append(base64_image)

        # 关闭buffer
        buffer.close()

    # 关闭PDF文档
    pdf_document.close()

    return base64_images


def transfer_image_to_text_dict(image):
    print(f"processing page {image['id']}")
    prompt = '''
Please convert the content of the image into Markdown text, following a logical reading order and ignore headers and footers. 
Use LaTeX for any formulas, equations, or chemical structures. 
For important illustrations, provide a detailed written description of their content. Ignore non-essential visuals. 
For blank pages, return the output as: 
```markdown
 
```
Ensure the conversion is clear, precise, and adheres to proper Markdown syntax.
'''

    max_retries = 2
    retries = 0

    while retries <= max_retries:
        try:
            request_params = {
                "model": args.model_name,
                "messages": [
                                {"role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{image['image']}",
                                            "detail": "high"
                                        },
                                    },
                                ]}
                            ]
            }
            
            # 如果模型名称包含 'gpt'，添加特殊参数
            if 'gpt' in args.model_name.lower():
                request_params.update({
                    "extra_body": {},
                    "extra_headers": {'apikey': args.api_key}
                })
            
            # Get OCR result using OpenAI client
            client = OpenAI(
                base_url=args.base_url,
                api_key=args.api_key,
                timeout=600
            )
            response = client.chat.completions.create(**request_params)
            result = response.choices[0].message.content
            print(result)
            result = clean_text(result)
            text_dict = {'id': image['id'], 'text': result}
            return text_dict  # 如果成功，返回response
        except Exception as e:
            retries += 1
            if retries > max_retries:
                print(f"Error: {e}")
                text_dict = {'id': image['id'], 'text': f'request error {e}'}
                return text_dict  # 超过重试次数，返回自定义错误信息
            time.sleep(10)  # 等待10秒后重试


def process_image(image):
    text = transfer_image_to_text_dict(image)
    return text


def transfer_images_to_text_dict_list(base64_images, process_num=1):
    images = [{'id': i, 'image': base64_image} for i, base64_image in enumerate(base64_images)]
    with Pool(processes=process_num) as pool:
        results = pool.map(process_image, images)
    return results


def transfer_PDF_to_text_dict_list(pdf_file_path, process_num=1):
    images = transfer_pdf_to_base64_images(pdf_file_path)
    text_dict_list = transfer_images_to_text_dict_list(images, process_num)
    return text_dict_list


if __name__ == "__main__":
    # 使用argparse接收文件夹参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        type=str,
                        required=True,
                        help='The path of the folder containing the PDF files')
    parser.add_argument('--model_name',
                        type=str,
                        required=True,
                        help='Name of the model to use')
    parser.add_argument('--base_url',
                        default='http://localhost:8998/v1',
                        help='Base URL for the API endpoint')
    parser.add_argument('--api_key',
                        default='dummy',
                        help='API key for the model service')
    parser.add_argument('--num_workers',
                        type=int,
                        default=1,
                        help='Number of worker processes to split the PDF files')
    parser.add_argument('--worker_id',
                        type=int,
                        default=0,
                        help='Worker ID (0-based index) for this process')
    parser.add_argument('--process_num',
                        type=int,
                        default=32,
                        help='Number of processes to use in the process pool')
    args = parser.parse_args()
    
    pdf_path_list = find_files(args.data_path, '*.pdf')
    # sort pdf_path_list by name
    pdf_path_list.sort()
    print(f"Found {len(pdf_path_list)} PDF files")
    
    # 分割PDF列表给当前worker处理
    if args.num_workers > 1:
        # 计算当前worker应处理的PDF文件
        worker_pdfs = [pdf for i, pdf in enumerate(pdf_path_list) 
                      if i % args.num_workers == args.worker_id]
        print(f"Worker {args.worker_id} processing {len(worker_pdfs)}/{len(pdf_path_list)} PDFs")
    else:
        worker_pdfs = pdf_path_list
    
    for pdf_path in tqdm(worker_pdfs):
        # 检查是否已经存在结果文件
        result_file = pdf_path[:-3] + 'jsonl'
        if os.path.exists(result_file):
            try:
                # 读取已有的结果文件
                existing_results = load_jsonl(result_file)
                # 检查request error出现的次数
                error_count = sum(1 for item in existing_results if "request error" in item['text'])
                
                # 如果error次数小于5，认为已经处理好了，跳过该pdf
                if error_count < 5:
                    print(f'Skipping {pdf_path}: already processed with {error_count} errors')
                    continue
                else:
                    print(f'Reprocessing {pdf_path}: found {error_count} errors in existing results')
            except Exception as e:
                print(f'Error checking existing results for {pdf_path}: {e}. Will reprocess.')
        text_dict_list = transfer_PDF_to_text_dict_list(pdf_path, args.process_num)
        print(f'Processed {pdf_path}')
        write_jsonl(pdf_path[:-3] + 'jsonl', text_dict_list)

