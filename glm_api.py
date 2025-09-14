from zai import ZhipuAiClient
import base64
import os
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models
os.environ['GOOGLE_APPLICATION_CREDENTIALS']='./gemini_exp.json'
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

def evaluate(image_urls):
    prompt="""
**Act as a professional image quality and identity evaluation system. You will receive one reference image (the first one) followed by multiple generated images (others) for assessment. For each generated image, evaluate based on these criteria:**

1.  **Structural Integrity and Reasonableness (40% weight):** Assess the inherent rationality of the generated image itself. For human/animal faces: evaluate facial symmetry, proportional distribution of facial features, anatomical correctness, and natural appearance. For objects: evaluate structural coherence, physical plausibility, and absence of deformities or artifacts.

2.  **Identity Faithfulness to Reference (60% weight):** Determine the degree to which the person/object in the generated image is the same as in the reference image. Consider facial features, distinctive characteristics, and overall likeness for persons; consider form, texture, and defining attributes for objects.

**Scoring Guidelines:**
- Provide a single score from 1 to 100 for each generated image, where a higher score indicates a better quality image that is more faithful to the reference.
- **Ensure meaningful score distribution:** Apply strict grading with significant variance (e.g., 50-100 range) to clearly differentiate between excellent, good, average, and poor results. Avoid score compression.Please ensure the average score is 80.
- Output **only** a Python list of numerical scores (e.g., `[85, 72, 78, 95, 70]`) with no additional text, explanations, or formatting.
"""
    vertexai.init(project="mmu-gemini-caption-1-5pro", location="us-central1")
    model = GenerativeModel("gemini-2.5-pro")
    def generate(filenames):
        """
        Generates a description for an image file using the Gemini model.
        """
        contents=[]
        for filename in filenames:
            with open(filename, "rb") as f:
                image_content = f.read()
            image_file = Part.from_data(image_content, mime_type="image/png") 
            contents.append(image_file)
        
        contents.append(prompt)
        responses = model.generate_content(
            contents, generation_config=generation_config, safety_settings=safety_settings,
        )
        return responses.text

    generation_config = { "max_output_tokens": 2048, "temperature": 1e-5, "top_p": 1.0}
    safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }
    return generate(image_urls)


def glm_evaluate(image_paths):
    client = ZhipuAiClient(api_key="66b96eb48a244f08835e7c7b02e04610.C8DhlMDgoGJuoJmh")
    prompt="""
**Act as a professional image quality and identity evaluation system. You will receive one reference image (the first one) followed by multiple generated images (others) for assessment. For each generated image, evaluate based on these criteria:**

1.  **Structural Integrity and Reasonableness (40% weight):** Assess the inherent rationality of the generated image itself. For human/animal faces: evaluate facial symmetry, proportional distribution of facial features, anatomical correctness, and natural appearance. For objects: evaluate structural coherence, physical plausibility, and absence of deformities or artifacts.

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
def extract(text,cla=None):
    prompt=f'''
    Please extract the information about the {cla} described in the text: {text}.
    Output them in a simple sentence less than 20 words.
    If there is no information that describes {cla} ,then give an empty output.
    Notice: Do not output any other information.
    '''
    vertexai.init(project="mmu-gemini-caption-1-5pro", location="us-central1")
    model = GenerativeModel("gemini-2.5-pro")
    generation_config = { "max_output_tokens": 2048, "temperature": 1e-5, "top_p": 1.0}
    safety_settings = {
        generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }
    contents=[]
    contents.append(prompt)
    responses = model.generate_content(
        contents, generation_config=generation_config, safety_settings=safety_settings,
    )
    return responses.text
def glm_extract(text,cla=None):
    client = ZhipuAiClient(api_key="66b96eb48a244f08835e7c7b02e04610.C8DhlMDgoGJuoJmh")
    prompt=f'''
    Please extract the information about the {cla} described in the text: {text}.
    Output them in a simple sentence less than 20 words.
    If there is no information that describes {cla} ,then give an empty output.
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
    # print(evaluate(["/share/project/emllm_mnt.1d/mnt/hpfs/baaiei/daigaole/code/UnicR1/showo/tmp_result/best_image/adrien_brody/image_000004.png"]))
    print(extract('<adrien_brody>\'s name is <adrien_brody>','man'))
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


