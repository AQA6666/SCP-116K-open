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
            response = request_one_turn_with_one_image(prompt, image['image'])
            print(response)
            if 'sorry' in response or 'unable' in response or '抱歉' in response:
                raise Exception(response)
            if not response.startswith('```markdown'):
                if '```markdown\n' in response:
                    response = response[response.index('```markdown\n'):]
                elif '```markdown' in response:
                    response = response[response.index('```markdown'):]
            response = clean_text(response)
            text_dict = {'id': image['id'], 'text': response}
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


def transfer_images_to_text_dict_list(base64_images):
    images = [{'id': i, 'image': base64_image} for i, base64_image in enumerate(base64_images)]
    with Pool() as pool:
        results = pool.map(process_image, images)
    return results


def transfer_PDF_to_text_dict_list(pdf_file_path):
    images = transfer_pdf_to_base64_images(pdf_file_path)
    text_dict_list = transfer_images_to_text_dict_list(images)
    return text_dict_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert PDF files to text using GPT-4o')
    parser.add_argument('source_file_dir', type=str, help='Directory containing PDF files to process')
    args = parser.parse_args()

    pdf_path_list = find_files(args.source_file_dir, '*.pdf')
    # pdf_path_list = random.sample(pdf_path_list, 4)
    processed_pdf_path_list = load_jsonl('processed_pdf_list.jsonl')
    pdf_path_list = [item for item in pdf_path_list if item not in processed_pdf_path_list]
    print(pdf_path_list)
    for pdf_path in tqdm(pdf_path_list):
        text_dict_list = transfer_PDF_to_text_dict_list(pdf_path)
        print(f'Processed {pdf_path}')
        processed_pdf_path_list.append(pdf_path)
        write_jsonl(pdf_path[:-3] + 'jsonl', text_dict_list)
        write_jsonl('processed_pdf_list.jsonl', processed_pdf_path_list)

