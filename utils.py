import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pdata import image_transform
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import json
# import deepspeed
from datasets import Dataset, IterableDataset
from packaging import version
from models import Showo, MAGVITv2, get_mask_chedule
import shutil
import copy
import random
import re
from typing import List, Any
import logging
from torch.distributed import get_rank
import torchvision.transforms as transforms
from feat.identity_detectors.facenet.facenet_model import InceptionResnetV1
from facenet_pytorch import MTCNN, InceptionResnetV1
# from torch.serialization import safe_globals
def remove_token(raw_str):
    pattern = r'<token_\d+>'

    # 替换为空字符串（去除匹配的标签）
    cleaned_str = re.sub(pattern, "", raw_str)

    return cleaned_str
def get_image_files(directory):
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    image_files = []
    
    # 检查目录是否存在
    if not os.path.exists(directory):
        raise ValueError(f"目录不存在: {directory}")
    if not os.path.isdir(directory):
        raise ValueError(f"路径不是目录: {directory}")
    
    # 只遍历当前目录，不递归子文件夹
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        # 只处理文件，忽略子文件夹
        if os.path.isfile(file_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file_path)
    
    return image_files
    
    return image_files

def check_embedding_dtype(model, input_ids, target_dtype):
    # 获取嵌入层
    embed_layer = model.showo.get_input_embeddings()
    # 生成嵌入向量并检查dtype
    embed_output = embed_layer(input_ids)
    assert embed_output.dtype == target_dtype, \
        f"嵌入层输出精度错误：期望 {target_dtype}，实际 {embed_output.dtype}"
    logging.debug("嵌入层输出精度验证通过")
def check_dtype(original_model,target_dtype):
    for name, param in original_model.named_parameters():
        if name.startswith("vision_model") or name.startswith("aligner") or name.startswith("gen"):
            if param.dtype != target_dtype:
                param.data = param.data.to(target_dtype)
                logging.warning(f"冻结层 {name} 精度不匹配，已强制转换为 {target_dtype}")
    for name, buf in original_model.named_buffers():
        if buf.dtype.is_floating_point:
            if buf.dtype != target_dtype:
                buf.data = buf.data.to(target_dtype)
    return original_model
from pathlib import Path
import os

def mkdir(path):
    folder_path = Path(path)
    # 1. 若路径不存在：直接创建文件夹（含父目录）
    if not folder_path.exists():
        folder_path.mkdir(parents=True, exist_ok=False)  # exist_ok=False 确保只创建不存在的路径
    # 2. 若路径存在，但不是文件夹（是文件）：先删除文件，再创建文件夹
    elif not folder_path.is_dir():
        os.remove(folder_path)  # 删除同名文件
        folder_path.mkdir(parents=True, exist_ok=False)  # 重新创建文件夹
    # 3. 若路径存在且是文件夹：直接跳过（无需操作）
    else:
        pass  # 文件夹已存在，无需处理
def read_json_to_dict(file_path):
    """
    读取JSON文件并转换为字典
    
    参数:
        file_path (str): JSON文件的路径
        
    返回:
        dict: 解析后的字典数据，如果文件不存在或解析出错则返回None
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            # 加载JSON数据并转换为字典
            data_dict = json.load(file)
            return data_dict
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 不存在")
        return None
    except json.JSONDecodeError:
        print(f"错误: 文件 '{file_path}' 不是有效的JSON格式")
        return None
    except Exception as e:
        print(f"读取文件时发生错误: {str(e)}")
        return None


def save_distributed_model(
    model, 
    optimizer, 
    save_dir, 
    epoch=0, 
):
    from torch.distributed import get_rank
    if get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)
        # 核心修改：移除 pytorch_version 字段（避免保存TorchVersion类型）
        save_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None
            # 删掉这行："pytorch_version": torch.__version__
        }
        save_path = os.path.join(save_dir, f"model_epoch{epoch}.pt")
        torch.save(save_data, save_path)
        print(f"[rank0] DDP模型保存至：{save_path}")



def load_distributed_model(
    model, 
    optimizer=None, 
    save_dir=None, 
    device="cuda"
):
    """
    修复PyTorch 2.6+ 安全加载问题 + 分布式适配
    """
    # 1. 检查保存目录
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"模型保存目录不存在：{save_dir}")

    # ---------------------- 普通DDP加载（核心修复：处理weights_only问题） ----------------------
    model_files = [
        f for f in Path(save_dir).iterdir() 
        if f.is_file() and f.name.startswith("model_epoch") and f.name.endswith(".pt")
    ]
    if not model_files:
        raise FileNotFoundError(f"{save_dir} 中无DDP模型文件（需以model_epoch开头，.pt结尾）")
    
    latest_model_file = max(model_files, key=lambda x: int(x.name.split("_epoch")[-1].split(".")[0]))
    print(f"[rank{get_rank()}] DDP加载最新文件：{latest_model_file}")

    # 关键修复：用safe_globals上下文允许加载torch.torch_version.TorchVersion
    load_data = torch.load(
        latest_model_file,
        map_location=device,
        weights_only=True
    )
    
    # 加载模型参数（原始模型，后续DDP包装）
    model.load_state_dict(load_data["model_state_dict"])
    
    # 恢复优化器
    if optimizer and "optimizer_state_dict" in load_data and load_data["optimizer_state_dict"]:
        optimizer.load_state_dict(load_data["optimizer_state_dict"])
        print(f"[rank{get_rank()}] 优化器状态恢复完成")
        
        epoch = load_data["epoch"]
        print(f"[rank{get_rank()}] DDP模型加载完成（epoch {epoch}）")

    # 确保模型在目标设备
    model.to(device)
    return model, optimizer, epoch

def load_single_model_weights_from_file(
    model, 
    weight_file_path, 
    device="cuda"
):
    # 1. 基础校验：检查权重文件是否存在
    if not os.path.exists(weight_file_path):
        raise FileNotFoundError(f"模型权重文件不存在：{weight_file_path}")

    # 2. 打印加载信息
    print(f"正在加载模型权重文件：{weight_file_path}")
    load_data = torch.load(
        weight_file_path,
        map_location=device,
        weights_only=True
    )

    # 4. 验证权重文件核心结构（必须包含模型参数和轮次信息）
    required_keys = ["model_state_dict", "epoch"]
    for key in required_keys:
        if key not in load_data:
            raise KeyError(
                f"权重文件 {weight_file_path} 结构不合法，缺少必要键：{key}\n"
                "请确保权重文件是通过 'torch.save({\"model_state_dict\": model.state_dict(), \"epoch\": epoch, ...})' 保存的"
            )

    # 5. 加载模型参数
    model.load_state_dict(load_data["model_state_dict"])
    print(f"模型权重加载完成！对应训练轮次：epoch {load_data['epoch']}")

    # 6. 确保模型移动到目标设备（避免加载后设备不匹配）
    model.to(device)

    # 返回加载后的模型和训练轮次
    return model, load_data["epoch"]
import math

def calculate_distance(list1, list2):
    """计算两个列表的欧氏距离"""
    if len(list1) != len(list2):
        raise ValueError("两个列表的长度必须相同")
    
    squared_diff_sum = 0.0
    for a, b in zip(list1, list2):
        squared_diff_sum += (a - b) **2
    
    return squared_diff_sum
def calculate_bleu(reference, candidate):
    # 将字符串转换为字符列表（适用于中文）
    reference_tokens = list(reference)
    candidate_tokens = list(candidate)
    
    # 使用平滑函数和较低阶的n-gram
    smoothie = SmoothingFunction().method1
    bleu_score = sentence_bleu([reference], candidate, 
                weights=(0.7, 0.3), # 只使用1-gram到3-gram
                smoothing_function=smoothie)
    
    return bleu_score

def extract_list_from_response(response_text: str) -> List[Any]:
    """
    从大模型回答中提取列表并解析成Python列表
    
    Args:
        response_text: 大模型的回答文本
        
    Returns:
        List[Any]: 解析后的列表，如果找不到列表则返回空列表
    """
    # 尝试多种方式提取列表
    
    # 1. 尝试解析JSON格式的列表
    try:
        # 查找类似 ["item1", "item2"] 或 [1, 2, 3] 的JSON列表
        json_pattern = r'\[.*?\]'
        json_matches = re.findall(json_pattern, response_text, re.DOTALL)
        if json_matches:
            # 取最后一个匹配（通常是最完整的）
            return json.loads(json_matches[-1])
    except:
        pass
    
    # 2. 尝试提取带编号的列表 (1. item1, 2. item2, ...)
    numbered_pattern = r'(?:\d+[\.\)]|\-|\*)\s*([^\n]+)'
    numbered_items = re.findall(numbered_pattern, response_text)
    if numbered_items:
        return [item.strip() for item in numbered_items]
    
    # 3. 尝试提取带符号的列表 (- item1, * item2, • item3)
    bullet_pattern = r'(?:[\-\*•])\s*([^\n]+)'
    bullet_items = re.findall(bullet_pattern, response_text)
    if bullet_items:
        return [item.strip() for item in bullet_items]
    
    # 4. 尝试提取换行分隔的列表项
    line_items = []
    for line in response_text.split('\n'):
        line = line.strip()
        # 跳过空行和明显不是列表项的行
        if line and not line.startswith(('当然', '好的', '以下是', '```')):
            # 检查是否是合理的列表项（有一定长度且不是完整的句子）
            if 2 <= len(line) <= 100 and not line.endswith(('。', '!', '?')):
                line_items.append(line)
    
    if len(line_items) >= 2:  # 至少有两个项才认为是列表
        return line_items
    
    return []  # 没有找到列表

def extract_and_clean_list(response_text: str) -> List[str]:
    """
    提取并清理列表，去除多余的空格和标点
    """
    raw_list = extract_list_from_response(response_text)
    
    cleaned_list = []
    for item in raw_list:
        if isinstance(item, str):
            # 清理字符串项
            item = item.strip()
            # 去除开头的编号或符号
            item = re.sub(r'^[\d\-\.\*•\)\s]+', '', item)
            # 去除末尾的标点
            item = re.sub(r'[\.\!\,\;\:]$', '', item)
            if item:  # 只添加非空项
                cleaned_list.append(item)
        else:
            # 对于非字符串项（数字等），直接添加
            cleaned_list.append(item)
    
    return cleaned_list


def get_image_path(path):
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    folder_path=path
    image_files = []
    for file in os.listdir(folder_path):
        file_path = Path(folder_path) / file
        if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
            stem = file_path.stem
            if stem.isdigit():
                image_files.append((int(stem), file_path))
    
    if not image_files:
        return 0, None, None
    
    image_files.sort(key=lambda x: x[0])

    numbers = [num for num, _ in image_files]
    expected_numbers = list(range(len(image_files)))
    
    selected_num, selected_path = random.choice(image_files)
    
    return len(image_files), str(selected_path), selected_num

def get_questions(n):
    file_path='/home/daigaole/code/ex/dataset/unictokens_data/concept/train/adrien_brody/conversations.json'
    d={}
    with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    print('--------',n)
    for name in data.keys():
        print('----',name)
        if str(n) in name:
            d=random.choice(data[name])
    t=random.choice(data['text_only'])
    print('vqa',d)
    print('qa',t)
    return d,t

def normalize_logits(logits, eps=1e-8):
    mean = logits.mean(dim=-1, keepdim=True)
    std = logits.std(dim=-1, keepdim=True)
    return (logits - mean) / (std + eps) 


def extract_single_number(text):
    match = re.fullmatch(r'\s*-?\d+\.?\d*\s*', text.strip())
    if match:
        num_str = match.group().strip()
        return float(num_str) if '.' in num_str else int(num_str)
    return 0.5


def manage_top_images(image_path, score,  folder_path,top_n=30, counter_file="counter.txt"):
    """
    维护一个文件夹，只保留分数最高的n张图片，并更新score.json
    
    参数:
        image_path: 输入图片的路径
        score: 该图片的分数
        top_n: 保留的最高分数图片数量
        folder_path: 存储图片和score.json的文件夹路径
        counter_file: 用于保存计数器的文件名
    """
    # 确保文件夹存在
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    
    # 加载或初始化计数器
    counter_path = os.path.join(folder_path, counter_file)
    if os.path.exists(counter_path):
        with open(counter_path, 'r') as f:
            try:
                counter = int(f.read().strip())
            except:
                counter = 0  # 如果计数器文件损坏，重置为0
    else:
        counter = 0
    
    # 加载或初始化分数记录
    score_json_path = os.path.join(folder_path, "score.json")
    if os.path.exists(score_json_path):
        with open(score_json_path, 'r') as f:
            try:
                score_dict = json.load(f)
            except:
                score_dict = {}  # 如果JSON文件损坏，重置为空
    else:
        score_dict = {}
    
    # 检查当前图片数量是否小于top_n，或者分数足够高
    current_images = len(score_dict)
    should_add = False
    
    if current_images < top_n:
        should_add = True
    else:
        # 计算当前最低分
        min_score = min(score_dict.values())
        if score > min_score:
            should_add = True
    
    if should_add:
        # 如果需要，删除最低分的图片
        if current_images >= top_n:
            # 找到最低分的图片(可能有多个，删除第一个)
            min_score = min(score_dict.values())
            for img_name, img_score in list(score_dict.items()):
                if img_score == min_score:
                    # 删除图片文件
                    img_path = os.path.join(folder_path, img_name)
                    if os.path.exists(img_path):
                        os.remove(img_path)
                    # 从分数字典中移除
                        del score_dict[img_name]
                    break
        
        # 生成新的图片名称
        counter += 1
        new_img_name = f"image_{counter:06d}.png"
        new_img_path = os.path.join(folder_path, new_img_name)
        
        # 复制图片
        shutil.copy2(image_path, new_img_path)
        
        # 更新分数字典
        score_dict[new_img_name] = score
        
        # 保存更新后的计数器
        with open(counter_path, 'w') as f:
            f.write(str(counter))
        
        # 保存更新后的分数记录
        with open(score_json_path, 'w') as f:
            json.dump(score_dict, f, indent=4)
    
    return should_add  # 返回是否成功添加了图片

import face_recognition

def face_recognition_score(generated_image_path,data_root,concept):
    generated_image = face_recognition.load_image_file(generated_image_path)
    generated_encoding = face_recognition.face_encodings(generated_image)[0]
    path=os.path.join(data_root,'concept/train',concept)
    paths=get_image_files(path)
    scores=[]
    for p in paths:
        try:
            known_image = face_recognition.load_image_file(p)
            known_encoding = face_recognition.face_encodings(known_image)[0]
            # 计算人脸距离（越小越相似）
            distance = face_recognition.face_distance([known_encoding], generated_encoding)[0]
            
            # 将距离转换为0-100的分数（距离0→100分，距离0.6→0分）
            max_distance = 1  # 默认阈值
        except:
            continue
        if distance >= max_distance:
            s=0.0
        s = (1 - distance / max_distance)
        if s<0.5:
            s=0
        else:
            s=(s-0.5)*2
        scores.append(s)
    if len(scores)>0:
        score=sum(scores)/len(scores)
    else:
        raise RuntimeError("facerecogition评分失败")
    return score


device = 'cuda:3'
identity_detector = InceptionResnetV1(
    pretrained=None,
    classify=False,
    num_classes=None,
    dropout_prob=0.6,
    device=device,
)
identity_detector.logits = nn.Linear(512, 8631)
identity_model_file='./facenet_20180402_114759_vggface2.pth'
identity_detector.load_state_dict(torch.load(identity_model_file, map_location=device))
identity_detector.eval()
identity_detector.to(device)


mtcnn = MTCNN(
    image_size=160,  # 对齐后的人脸尺寸（与 Facenet 输入匹配）
    margin=32,       # 裁剪时的边缘余量
    min_face_size=20,  # 最小可检测人脸尺寸
    thresholds=[0.6, 0.7, 0.7],  # 三级网络的置信度阈值
    factor=0.709,    # 图像金字塔缩放因子
    post_process=True,  # 对齐后是否标准化
    device=device  # 设备
)

def facenet_score(image_path1,data_root,concept):
    img1 = Image.open(image_path1).convert('RGB')
    face_tensor1 = mtcnn(img1)
    with torch.no_grad():  # 关闭梯度计算，节省内存并加速
        identity_embeddings1 = identity_detector(face_tensor1.unsqueeze(0).to(device))
    path=os.path.join(data_root,'concept/train',concept)
    paths=get_image_files(path)
    scores=[]
    for p in paths:
        # print(p)
        try:
            img2 = Image.open(p).convert('RGB')
            face_tensor2 = mtcnn(img2)
            with torch.no_grad():
                identity_embeddings2 = identity_detector(face_tensor2.unsqueeze(0).to(device))
            # print(f"distance: {torch.norm(identity_embeddings1 - identity_embeddings2)}")
            # distance=torch.norm(identity_embeddings1 - identity_embeddings2)
            cos_sim = torch.nn.functional.cosine_similarity(identity_embeddings1, identity_embeddings2)  # 相似度
            if cos_sim<0.5:
                cos_sim=0
            else:
                cos_sim=2*(cos_sim-0.5)
            s=cos_sim
            scores.append(s)
        except:
            continue
    if len(scores)>0:
        score=sum(scores)/len(scores)
    else:
        raise RuntimeError("facenet评分失败")
    return score
    # print(f"cosine distance",cos_dist)
# print(face_recognition_score('/share/project/emllm_mnt.1d/mnt/hpfs/baaiei/daigaole/code/UnicR1/showo/ref_image/adrien_brody/E0B4R0G1N0.png','/share/project/emllm_mnt.1d/mnt/hpfs/baaiei/daigaole/code/UnicR1/dataset/unictokens_data','adrien_brody'))
