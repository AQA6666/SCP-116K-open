import base64
import openai
import time
from PIL import Image
import io


def request_one_turn(prompt, model="gpt-4o", temperature=1.0, top_p=1.0, max_tokens=16384):
    messages = [
        {"role": "user", "content": prompt},
    ]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def request_one_turn_with_one_image(prompt, base64_image, model="gpt-4o"):
    messages = [
        {"role": "user",
         "content": [
             {"type": "text", "text": prompt},
             {
                 "type": "image_url",
                 "image_url": {
                     "url": f"data:image/jpeg;base64,{base64_image}",
                     "detail": "high"
                 },
             },
         ]}
    ]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
    )

    return response.choices[0].message.content


def request_one_turn_with_images(prompt, base64_images, model="gpt-4o"):
    # Construct the content list starting with the text prompt
    content = [{"type": "text", "text": prompt}]
    
    # Add each image as an image_url entry
    for base64_image in base64_images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "high"
            }
        })
    
    # Create the messages structure
    messages = [{"role": "user", "content": content}]
    
    # Make the API call
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
    )

    return response.choices[0].message.content


def request_with_messages(messages, model="gpt-4o"):
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
    )

    return response.choices[0].message.content


def safe_request_one_turn(prompt, error_response='error', model="gpt-4o", retry_lapse=30):
    max_retries = 2
    retries = 0

    while retries <= max_retries:
        try:
            response = request_one_turn(prompt, model=model)
            return response  # 如果成功，返回response
        except Exception as e:
            retries += 1
            if retries > max_retries:
                print(e)
                return error_response  # 超过重试次数，返回自定义错误信息
            time.sleep(retry_lapse)  # 等待重试


def encode_and_resize_image(image_path, scale_factor=2):
    """
    读取图片，按比例放大，并返回base64编码
    Args:
        image_path: 图片路径
        scale_factor: 放大倍数，默认2倍
    Returns:
        base64编码的图片字符串
    """
    # 打开图片
    with Image.open(image_path) as img:
        # 计算新的尺寸
        new_width = int(img.width * scale_factor)
        new_height = int(img.height * scale_factor)
        # 调整图片大小，使用LANCZOS重采样以获得更好的质量
        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # 将图片保存到内存中的字节流
        buffer = io.BytesIO()
        resized_img.save(buffer, format=img.format or 'PNG')
        # 获取字节并转换为base64
        return base64.b64encode(buffer.getvalue()).decode('utf-8')


if __name__ == '__main__':
    print(request_one_turn('已知 $(1+2i)(a+i)$ 的实部与虚部互为相反数，则实数 $a=$', model='gpt-4o'))
