import pdb
import base64
from openai import OpenAI
from pathlib import Path
import json
# import google.generativeai as genai
import requests
import PIL.Image
from tqdm import tqdm
import re
from collections import deque
from PIL import Image
import cv2
# GPT Client
client = OpenAI(
    api_key='sk-YhR1FycY6wzVIwSaAbC3FaE8571141A29aCb14E4A27dAc8b',
    base_url="https://api.gptplus5.com/v1"   # 设置代理 URL
    # base_url="https://api.openai-proxy.org/v1"
)

def encode_image_to_base64(image_path, return_resolution=False):
    """
    将图片转换为 base64 编码
    如果图像宽度大于2048像素，会等比例缩放至宽度不超过2048
    
    Args:
        image_path: 图片路径
        return_resolution: 是否返回分辨率
        
    Returns:
        base64编码字符串, 或 (base64编码字符串, (宽度, 高度))
    """
    # 打开图像检查尺寸
    with Image.open(image_path) as img:
        original_width, original_height = img.size
        
        # 检查是否需要缩放
        if original_width > 2048:
            # 计算缩放比例
            scale_factor = 2048 / original_width
            new_width = 2048
            new_height = int(original_height * scale_factor)
            
            # 缩放图像
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 将缩放后的图像保存到内存中
            import io
            buffer = io.BytesIO()
            resized_img.save(buffer, format=img.format or 'JPEG')
            buffer.seek(0)
            
            # 编码为base64
            base64_str = base64.b64encode(buffer.read()).decode('utf-8')
            
            if return_resolution:
                return base64_str, (new_width, new_height)
        else:
            # 不需要缩放，直接读取文件进行编码
            with open(image_path, "rb") as image_file:
                base64_str = base64.b64encode(image_file.read()).decode('utf-8')
            
            if return_resolution:
                return base64_str, (original_width, original_height)
    
    return base64_str

def chat_with_images_gpt(prompt, image_paths, model_name="gpt-4o-mini"):#"gemini-2.5-pro-exp-03-25"):
    """
    使用 GPT-4 Vision 模型处理文本和图片，带有指数退避的重试机制
    """
    import time
    import random
    import logging

    # 配置参数
    max_retries = 5  # 最大重试次数
    base_delay = 5   # 基础延迟（秒）
    max_delay = 60   # 最大延迟（秒）
    jitter = 0.1     # 随机抖动因子

    # 可重试的错误类型
    RETRYABLE_ERRORS = [
        "rate_limit",
        "timeout",
        "connection_error",
        "server_error",
        "500",
        "502",
        "503",
        "504"
    ]

    for attempt in range(max_retries):
        try:
            # 准备消息内容
            content = [{"type": "text", "text": prompt}]
            
            # 添加所有图片
            for image_path in image_paths:
                base64_image = encode_image_to_base64(image_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })

            # 创建聊天完成
            # pdb.set_trace()
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_tokens=4096
            )
            # pdb.set_trace()
            return response.choices[0].message.content
        
        except Exception as e:
            error_msg = str(e).lower()
            
            # 判断是否是可重试的错误
            should_retry = any(err in error_msg for err in RETRYABLE_ERRORS)
            
            if should_retry and attempt < max_retries - 1:
                # 计算延迟时间（指数退避 + 随机抖动）
                delay = min(base_delay * (2 ** attempt), max_delay)
                jitter_amount = delay * jitter
                actual_delay = delay + random.uniform(-jitter_amount, jitter_amount)
                
                logging.warning(
                    f"请求失败 (尝试 {attempt + 1}/{max_retries})\n"
                    f"错误: {error_msg}\n"
                    f"等待 {actual_delay:.2f} 秒后重试..."
                )
                
                time.sleep(actual_delay)
                continue
            else:
                # 最后一次尝试失败或不可重试的错误
                error_type = "最后一次尝试失败" if attempt == max_retries - 1 else "不可重试的错误"
                logging.error(f"{error_type}: {error_msg}")
                return f"API调用失败 ({error_type}): {error_msg}"

# encoded_image = encode_image_to_base64('/home/daigaole/code/ex/showo_feat/tmp_result/1_ref.png')
# print(len(encoded_image))
# print(chat_with_images_gpt("How much do you think the person is the same one in two images?\nPlease use a number ranging from 0 to 1 to represent.\nPlease only output a number.\n",['/home/daigaole/code/ex/showo_feat/tmp_result/1_ref.png','/home/daigaole/code/ex/showo_feat/tmp_result/0_ref.png']))