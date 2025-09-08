from zai import ZhipuAiClient
import base64

def image_to_base64(image_path):
    """
    将本地图片文件编码为 Base64 字符串
    :param image_path: 本地图片路径（如 "test.png"、"./images/photo.jpg"）
    :return: Base64 编码字符串（UTF-8 格式）
    """
    try:
        # 1. 以二进制模式读取图片
        with open(image_path, "rb") as image_file:
            # 2. 读取二进制数据并编码为 Base64（返回二进制字符串）
            base64_binary = base64.b64encode(image_file.read())
            # 3. 解码为 UTF-8 字符串（便于后续使用）
            base64_str = base64_binary.decode("utf-8")
        return base64_str
    except FileNotFoundError:
        return f"错误：图片路径 {image_path} 不存在"
    except Exception as e:
        return f"编码失败：{str(e)}"


def glm_evaluate(image_paths):
    client = ZhipuAiClient(api_key="85a57f3de9fb4d8fa9b34d63c9e2aba3.eTv6vtDpYukhYEWc")
    prompt="""
**Act as a professional image quality and identity evaluation system. You will receive one reference image followed by multiple generated images for assessment. For each generated image, evaluate based on these criteria:**

1.  **Structural Integrity and Reasonableness (40% weight):** Assess the inherent rationality of the generated image itself. For human faces: evaluate facial symmetry, proportional distribution of facial features, anatomical correctness, and natural appearance. For objects: evaluate structural coherence, physical plausibility, and absence of deformities or artifacts.

2.  **Identity Faithfulness to Reference (60% weight):** Determine the degree to which the person/object in the generated image is the same as in the reference image. Consider facial features, distinctive characteristics, and overall likeness for persons; consider form, texture, and defining attributes for objects.

**Scoring Guidelines:**
- Provide a single score from 1 to 100 for each generated image, where a higher score indicates a better quality image that is more faithful to the reference.
- **Ensure meaningful score distribution:** Apply strict grading with significant variance (e.g., 50-100 range) to clearly differentiate between excellent, good, average, and poor results. Avoid score compression.Please ensure the average score is 80.
- Output **only** a Python list of numerical scores (e.g., `[85, 72, 78, 95, 70]`) with no additional text, explanations, or formatting.
"""

    base=[]
    for path in image_paths:
        base.append(image_to_base64(path))
    tmp=[{
            "type": "image_url",
            "image_url": {
                "url": f"{i}"
            }
            } for i in base]
    tmp.append({
                "type": "text",
                "text": f"{prompt}"
                })
    # 创建聊天完成请求
    response = client.chat.completions.create(
        model="glm-4.1v-thinking-flash",
        stream= False,
        thinking= {
            "type": "enabled"
        },
        do_sample= False,
        temperature= 0,
        top_p= 0.9,
        messages= [
            {
            "role": "system",
            "content": "You are a picture scorer."
            },
            {
            "role": "user",
            "content": tmp
            }
        ],
        max_tokens= 4096
    )
    # print(response.choices[0].message.content)
    return response.choices[0].message.content

def glm_extract(text,question):
    client = ZhipuAiClient(api_key="85a57f3de9fb4d8fa9b34d63c9e2aba3.eTv6vtDpYukhYEWc")
    prompt=f'''
    The input for a small LLM is {question}.
    The output of the small LLM is {text}.
    Please recognize the valid part of {text} and output it(less than 20 words).
    Please notice: only output the valid part,do not output any extra info.
    If there is nothing valid,then give an empty output.
    ''' 
    prompt=f'''
    Please extract the information about <adrien_brody> in the text: {text}.
    Output them in a simple sentence less than 20 words.
    If there is no information about <adrien_brody>,then give an empty output.
    Notice: Do not output any other information.
    '''
    response = client.chat.completions.create(
        model="glm-4.1v-thinking-flash",
        stream= False,
        thinking= {
            "type": "enabled"
        },
        do_sample= False,
        temperature= 0.1,
        top_p= 0.95,
        messages= [
            {
            "role": "system",
            "content": "You are a text analyst."
            },
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": f"{prompt}"
                }
            ]
            }
        ],
        max_tokens= 4096
    )
    # print(response.choices[0].message.content)
    return response.choices[0].message.content

# ------------------- 调用示例 -------------------
if __name__ == "__main__":
    # 替换为你的本地图片路径
    # print(glm_evaluate(["/home/daigaole/code/ex/showo_feat/best_result/part_4.png","/home/daigaole/code/ex/showo_feat/best_result/part_5.png","/home/daigaole/code/ex/showo_feat/best_result/part_6.png"]))
    print(glm_extract('<adrien_brody>\'s name is <adrien_brody>','Please output something about <adrien_brody>'))
    exit()
    local_image_path = "/home/daigaole/code/ex/showo_feat/best_result/part_4.png"  # 支持 PNG、JPG、JPEG、BMP 等常见格式
    base64_result = image_to_base64(local_image_path)
    b1=image_to_base64("/home/daigaole/code/ex/showo_feat/best_result/part_5.png")
    print(b1)
    # 创建聊天完成请求
    response = client.chat.completions.create(
        model="glm-4.1v-thinking-flash",
        stream= False,
        thinking= {
            "type": "enabled"
        },
        do_sample= True,
        temperature= 0.1,
        top_p= 0.95,
        messages= [
            {
            "role": "system",
            "content": "You are a picture scorer."
            },
            {
            "role": "user",
            "content": [
                {
                "type": "image_url",
                "image_url": {
                    "url": f"{base64_result}"
                }
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"{b1}"
                }
                },
                {
                "type": "text",
                "text": "Please score each image and only output the scores"
                }
            ]
            }
        ],
        max_tokens= 4096
    )

    # 获取回复
    print(response.choices[0].message.content)
