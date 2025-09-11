import os
import textwrap
from glm_api import glm_extract,glm_evaluate
from collections import defaultdict
from typing import Any, Callable, Optional, Union
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from pdata import image_transform
import numpy as np
from pathlib import Path
import torch
import torch.utils.data
import json
import transformers
import deepspeed
from datasets import Dataset, IterableDataset
from packaging import version
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from clip_eval import SHOWO_P_CLIPEvaluator
from models import Showo, MAGVITv2, get_mask_chedule
from clip.model import build_model
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next, create_attention_mask_for_mmu
from transformers.integrations import is_deepspeed_zero3_enabled 
# from gpttest import chat_with_images_gpt
from api import evaluate,extract
from record_best import update_best_results
from torch.utils.checkpoint import checkpoint
from transformers import (

    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,

    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from deepspeed import zero
from transformers.utils import is_peft_available
from training.utils import get_config, flatten_omega_conf, mask_or_random_replace_tokens, AverageMeter
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url
import shutil

import copy
import random
import re
from typing import List, Any
import logging
tt=local_rank = int(os.environ.get("LOCAL_RANK", 0))
# 1. 初始化日志配置（仅需执行一次，通常在程序入口）
logging.basicConfig(
    level=logging.INFO,  # 日志级别：只记录 >= INFO 的信息（过滤 DEBUG）
    format="%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s",  # 日志格式
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler(f"{tt}"+"_record.log", encoding="utf-8")  # 输出到文件（utf-8避免中文乱码）
    ]
)
from torch.distributed import get_rank
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
    use_deepspeed=False
):
    from torch.distributed import get_rank
    if get_rank() == 0:
        os.makedirs(save_dir, exist_ok=True)

        if use_deepspeed:
            model.save_checkpoint(
                save_dir=save_dir,
                tag=f"epoch_{epoch}",
                save_optimizer_states=True
            )
            print(f"[rank0] DeepSpeed模型保存至：{os.path.join(save_dir, f'epoch_{epoch}')}")
        else:
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

import torch
import os
from torch.distributed import get_rank
from pathlib import Path
# 导入安全加载相关工具
from torch.serialization import safe_globals

def load_distributed_model(
    model, 
    optimizer, 
    save_dir, 
    use_deepspeed=False, 
    device="cuda"
):
    """
    修复PyTorch 2.6+ 安全加载问题 + 分布式适配
    """
    # 1. 检查保存目录
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"模型保存目录不存在：{save_dir}")

    if use_deepspeed:
        # ---------------------- DeepSpeed加载（不受weights_only影响） ----------------------
        checkpoint_dirs = [
            d for d in Path(save_dir).iterdir() 
            if d.is_dir() and d.name.startswith("epoch_")
        ]
        if not checkpoint_dirs:
            raise FileNotFoundError(f"{save_dir} 中无DeepSpeed checkpoint目录（需以epoch_开头）")
        
        latest_ckpt_dir = max(checkpoint_dirs, key=lambda x: int(x.name.split("_")[-1]))
        print(f"[rank{get_rank()}] DeepSpeed加载最新目录：{latest_ckpt_dir}")

        load_info = model.load_checkpoint(
            load_dir=save_dir,
            tag=latest_ckpt_dir.name,
            load_optimizer_states=True
        )
        epoch = load_info["epoch"]
        print(f"[rank{get_rank()}] DeepSpeed模型加载完成（epoch {epoch}）")

    else:
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
        with safe_globals([torch.torch_version.TorchVersion]):
            # 保持weights_only=True（安全模式），同时允许信任的类型
            load_data = torch.load(
                latest_model_file,
                map_location=device,
                weights_only=True  # 保留安全模式，仅允许信任类型
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
generate_prompt='''
You are asked to generate an image based on this prompt: "{}"
Provide a brief, precise visualization of all elements in the prompt. Your description should:
1. Include every object mentioned in the prompt
2. Specify visual attributes (color, number, shape, texture) if specified in the prompt
3. Clarify relationships (e.g., spatial) between objects if specified in the prompt
4. Be concise (50 words or less)
5. Focus only on what's explicitly stated in the prompt
6. Do not elaborate beyond the attributes or relationships specified in the prompt
Do not miss objects. Output your visualization directly without explanation: 
'''
def get_image_path():
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    folder_path='/home/daigaole/code/ex/dataset/unictokens_data/concept/train/adrien_brody'
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

_,IMAGE_PATH,SELECTED_NUM=get_image_path()
IMAGE_PATH='/home/daigaole/code/ex/dataset/unictokens_data/black_512x512.png'
VQA,QA=get_questions(SELECTED_NUM)
# '/home/daigaole/code/ex/dataset/unictokens_data/concept/train/adrien_brody/3.png'
def prepare_inputs_and_labels(
        mask_id,
        config,
        vq_model,
        uni_prompting,
        mask_schedule,
        pixel_values_or_image_ids: Union[torch.FloatTensor, torch.LongTensor],
        texts: Union[str, str],
        min_masking_rate: float = 0.0,
        is_train: bool = True,
):

    image_tokens = vq_model.get_code(pixel_values_or_image_ids)
    image_tokens = image_tokens + len(uni_prompting.text_tokenizer)

    # create MLM mask and labels
    input_ids, labels, loss_weight, mask_prob = mask_or_random_replace_tokens(
        image_tokens,
        mask_id,
        config,
        mask_schedule=mask_schedule,
        is_train=is_train,
    )
    input_ids, masks, labels = uni_prompting((texts, input_ids, labels), 't2i')

    return input_ids, labels, mask_prob, image_tokens
def clone_model(model: torch.nn.Module) -> torch.nn.Module:
    new_model = copy.deepcopy(model)
    new_model.load_state_dict(copy.deepcopy(model.state_dict()))
    new_model.to(next(model.parameters()).device)
    
    return new_model
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
class unic_grpo(Trainer):
    def __init__(
        self,
        model,
        # ref_model,
        reward_funcs,
        args,
        train_args,
        config,
        dataset,
        vq_model,
        uni_prompting,
        optimizer,
        tokenizer,
        # ref_tokenizer,
        peft_config: Optional["PeftConfig"] = None,
        attn_implementation: str = "flash_attention_2",
    ):

        # freeze all vision encoders
        for name, param in model.named_parameters():
            if name.startswith("vision_model") or name.startswith("aligner") or name.startswith("gen"): # choose whatever you like here
                param.requires_grad = False
        for name, param in vq_model.named_parameters():
            param.requires_grad = False

        self.local_rank = local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        # # Reference model
        self.deepspeed_enabled = train_args.deepspeed
        if self.deepspeed_enabled:
            self.model, self.optimizer, _, _ = deepspeed.initialize(
                model=model,
                optimizer=optimizer,
                config_params='/home/daigaole/code/ex/showo_feat/ds_config.json',
                model_parameters=filter(lambda p: p.requires_grad, model.parameters())
            )
            self.model.module.gradient_checkpointing_enable()
        else:
            self.model=model
            self.optimizer=optimizer
        self.num_generations = args.num_gen  # = G in the GRPO paper
        self.group=int(self.num_generations/args.num_gpus)
        self.num_generations=int(self.num_generations/self.group)
        self.beta = train_args.beta
        self._metrics = defaultdict(list)
        self.model_accepts_loss_kwargs = False
        self.dataset=dataset
        self.args=args
        self.config=config
        self.vq_model=vq_model
        self.uni_prompting=uni_prompting
        self.tokenizer=tokenizer
        self.beta=0
        self.info=read_json_to_dict('/home/daigaole/code/ex/dataset/unictokens_data/concept/train/adrien_brody/info.json')
        self.t2i_condition=read_json_to_dict('/home/daigaole/code/ex/dataset/unictokens_data/concept/train/adrien_brody/t2i_conditions.json')
        if self.local_rank==0:
            save_distributed_model(self.model,self.optimizer,'/home/daigaole/code/ex/showo_feat/result',epoch=0)
            self.model,self.optimizer,_=load_distributed_model(self.model,self.optimizer,'/home/daigaole/code/ex/showo_feat/result',device=self.args.device)

        # if self.deepspeed_enabled:
        #     from deepspeed import zero
        #     self.model, _ = deepspeed.initialize(model=model, config_params=train_args.deepspeed)
        #     self.ref_model, _ = deepspeed.initialize(model=self.ref_model, config_params=train_args.deepspeed, eval_mode=True)

        # # Reward functions
        # if not isinstance(reward_funcs, list):
        #     reward_funcs = [reward_funcs]
        # for i, reward_func in enumerate(reward_funcs):
        #     if isinstance(reward_func, str) and 'hps' in reward_func:
        #         reward_funcs[i] = HPSv2(args)
        #     elif isinstance(reward_func, str) and 'git' in reward_func:
        #         reward_funcs[i] = GIT(args)
        #     elif isinstance(reward_func, str) and 'gdino' in reward_func:
        #         reward_funcs[i] = GDino(args)
        #     elif isinstance(reward_func, str) and 'orm' in reward_func:
        #         reward_funcs[i] = ORM(args)
        #     else:
        #         reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
        #             reward_func, num_labels=1, **model_init_kwargs
        #         )
        # self.reward_funcs = reward_funcs

        # # Reward processing class
        # if reward_processing_classes is None:
        #     reward_processing_classes = [None] * len(reward_funcs)
        # elif not isinstance(reward_processing_classes, list):
        #     reward_processing_classes = [reward_processing_classes]
        # else:
        #     if len(reward_processing_classes) != len(reward_funcs):
        #         raise ValueError("The number of reward processing classes must match the number of reward functions.")

        # for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
        #     if isinstance(reward_func, PreTrainedModel):
        #         if reward_processing_class is None:
        #             reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
        #         if reward_processing_class.pad_token_id is None:
        #             reward_processing_class.pad_token = reward_processing_class.eos_token
        #         # The reward model computes the reward for the latest non-padded token in the input sequence.
        #         # So it's important to set the pad token ID to the padding token ID of the processing class.
        #         reward_func.config.pad_token_id = reward_processing_class.pad_token_id
        #         reward_processing_classes[i] = reward_processing_class
        # self.reward_processing_classes = reward_processing_classes


        
        # if self.beta != 0:
        #     if self.is_deepspeed_enabled:
        #         self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
        #     else:
        #         self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        # else:
        #     self.ref_model = None


    def _get_per_token_logps(self, model, input_embeds, text_ids, img_ids, attention_mask):
        def _get_per_token_logps_part(logits, input_ids):
            logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
            # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
            per_token_logps = []

            for logits_row, input_ids_row in zip(logits, input_ids):
                log_probs = logits_row.log_softmax(dim=-1)
                token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
                per_token_logps.append(token_log_prob)
            return torch.stack(per_token_logps)
        # here, we only compute either text or image loss, so ids of other one could be omitted
        if img_ids is not None:
            # compute logits for image tokens
            hidden_states = model.language_model(inputs_embeds=input_embeds, attention_mask=attention_mask, output_hidden_states=True).hidden_states  # (B, L, V)
            last_hidden_states = hidden_states[-1]
            # (text input id, image start token, image input id)
            # text_ids: text input id + image start token
            # img_ids: img_id (image token)
            image_logits = model.gen_head(last_hidden_states[:, -(img_ids.size(1)+1):, :]) # image prediction
            
            img_input_ids = torch.cat([img_ids.new_zeros(img_ids.size(0), 1), img_ids], dim=1) # cat a random one here, since it is not used in the loss calculation
            per_token_logps_img = _get_per_token_logps_part(image_logits, img_input_ids) # only calculate image loss
            return torch.cat([
                per_token_logps_img.new_zeros(
                    (per_token_logps_img.size(0), input_embeds.size(1) - per_token_logps_img.size(1) - 1)
                ), # the return length should be the input length minus 1 (the last token does not need predict)
                per_token_logps_img
            ], 
            dim=1)
        else: # only calculate text ids
            hidden_states = model.language_model(inputs_embeds=input_embeds, attention_mask=attention_mask, output_hidden_states=True).hidden_states  # (B, L, V)
            last_hidden_states = hidden_states[-1]
            text_logits = model.language_model.lm_head(last_hidden_states) 
            per_token_logps_text = _get_per_token_logps_part(text_logits, text_ids) 
            return per_token_logps_text

    def _prepare_inputs(self, inputs):
        return inputs

    def train(self,return_outputs=False, num_items_in_batch=None):
        counter=0
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # other setting
        self.model.config.mask_token_id = self.model.showo.get_input_embeddings().num_embeddings - 1
        self.model.mask_token_id = self.model.showo.get_input_embeddings().num_embeddings - 1
        # self.ref_model.config.mask_token_id = self.ref_model.showo.get_input_embeddings().num_embeddings - 1
        # self.ref_model.mask_token_id = self.ref_model.showo.get_input_embeddings().num_embeddings - 1
        mask_schedule = get_mask_chedule(self.config.training.get("mask_schedule", "cosine"))
        mask_id = self.model.mask_token_id
        mask_dtype = self.model.showo.get_input_embeddings().weight.dtype
        self.model.output_size = self.config.new_total_vocab
        # self.ref_model.output_size = self.config.new_total_vocab
        signal=False
        for epoch in range(self.args.epoch):
            print(f"Epoch {epoch+1}")
            loss_list = []
            if self.args.t2i_data:
                loss_t2i_list = []
            if self.args.mmu_data:
                loss_mmu_list = []
            counter=0
            for batch, batch_idx, dataloader_idx in tqdm(self.dataset):
            # for _ in range(100):
                torch.cuda.empty_cache()
                batch_size_t2i = batch["t2i_flow"]["images"].shape[0]
                # batch_size_t2i=1
                #realize t2i inference
                self.config.model.showo.llm_vocab_size = len(self.tokenizer) - 10
                self.config.generation_timesteps = 50
                self.config.guidance_scale = 5

                nums_new_token_i_stage_1 = self.args.nums_new_token_i_stage_1
                nums_new_token_i_stage_2 = self.args.nums_new_token_i_stage_2
                new_tokens_stage_1 = [f"<token_{i}>" for i in range(nums_new_token_i_stage_1)]
                new_tokens_stage_2 = [f"<token_{i}>" for i in range(nums_new_token_i_stage_1, nums_new_token_i_stage_1 + nums_new_token_i_stage_2)]

                # mti_ref=self.ref_model.config.mask_token_id
                # self.ref_model.config.mask_token_id = self.ref_model.showo.get_input_embeddings().num_embeddings - 1
                # mask_token_id = self.ref_model.showo.get_input_embeddings().num_embeddings - 1
                # image_tokens_infer = torch.ones((batch_size_t2i, self.config.model.showo.num_vq_tokens),
                #                         dtype=torch.long, device=self.args.device) * mask_token_id
                global_logits=[]
                global_id=[]
                reward_text=[]
                for group_id in range(self.group):
                    logging.info(f"gpu {self.local_rank},number {group_id}")
                    mti=self.model.config.mask_token_id
                    self.model.config.mask_token_id = self.model.showo.get_input_embeddings().num_embeddings - 1
                    mask_token_id = self.model.showo.get_input_embeddings().num_embeddings - 1
                    image_tokens_infer = torch.ones((batch_size_t2i, self.config.model.showo.num_vq_tokens),
                                            dtype=torch.long, device=self.args.device) * mask_token_id
                    
                    
                    save_dir='/home/daigaole/code/ex/showo_feat/tmp_result/'
                    condition='A photo of '
                    for token in new_tokens_stage_1:
                        condition+=token
                    for token in new_tokens_stage_2:
                        condition+=token
                    condition+='<adrien_brody>.\n'
                    # us_prompt="what is in the image?"
                    # us_prompt="Describe the person in this image using 3-5 concise descriptive adjectives focused on appearance, expression, and demeanor.Do not output extra information except these adjectives."
                    r=random.randint(1,len(self.t2i_condition["personalized_driven_generation"]))
                    question=self.t2i_condition["personalized_driven_generation"][r-1]
                    us_prompt=f'''
                    Below is some information about <adrien_brody> : {self.info['extra_info']}
                    Please make inferences based on the following prompt: {question}. 
                    If the prompt relates to a specific item from the aforementioned information list, 
                    output and only output that exact item. 
                    If the prompt does not relate to any item in the list, 
                    output nothing (i.e., an empty response).
                    '''
                    image_ori = Image.open(IMAGE_PATH).convert("RGB")
                    # tranforming the image to the required resolution
                    image = image_transform(image_ori, resolution = self.config.dataset.params.resolution).to(self.args.device)
                    image = image.unsqueeze(0)


                    
                    image_tokens_mmu = self.vq_model.get_code(image)
                    image_tokens = image_tokens_mmu + len(self.uni_prompting.text_tokenizer)
                    us_input = self.uni_prompting.text_tokenizer(['USER: ' + us_prompt + ' ASSISTANT:'])['input_ids']
                    us_input = torch.tensor(us_input).to(self.args.device)
                    us_input = torch.cat([
                        (torch.ones(us_input.shape[0], 1) * self.uni_prompting.sptids_dict['<|mmu|>']).to(self.args.device),
                        (torch.ones(us_input.shape[0], 1) * self.uni_prompting.sptids_dict['<|soi|>']).to(self.args.device),
                        image_tokens,
                        (torch.ones(us_input.shape[0], 1) * self.uni_prompting.sptids_dict['<|eoi|>']).to(self.args.device),
                        (torch.ones(us_input.shape[0], 1) * self.uni_prompting.sptids_dict['<|sot|>']).to(self.args.device),
                        us_input
                    ], dim=1).long()
                    us_mask = create_attention_mask_for_mmu(us_input.to(self.args.device),
                                                eoi_id=int(self.uni_prompting.sptids_dict['<|eoi|>']))
                    us_toks_list = self.model.mmu_generate(
                        us_input, 
                        attention_mask=us_mask,
                        top_k=5,
                        eot_token=self.uni_prompting.sptids_dict['<|eot|>'],
                    )
                    us_toks_list= torch.stack(us_toks_list).squeeze()[None]
                    more_prompt = self.uni_prompting.text_tokenizer.batch_decode(us_toks_list, skip_special_tokens=True)[0].strip()
                    
                    logging.info(f"Question:{question}")
                    logging.info(f"Answer:{more_prompt}")
                    reward_text.append(calculate_bleu(self.info['extra_info'][r-1],more_prompt))
                    logging.info(f"{self.local_rank}: Bleu score is {reward_text}")
                    more_prompt=glm_extract(more_prompt,us_prompt)
                    if '<|begin_of_box|>' in more_prompt:
                        more_prompt=more_prompt.replace('<|begin_of_box|>','')
                    if '<|end_of_box|>' in more_prompt:
                        more_prompt=more_prompt.replace('<|end_of_box|>','')
                    logging.info(f"After Extraction:{more_prompt}")
                    condition+=self.info['info']+more_prompt
                    conditions = [condition] * batch_size_t2i
                    del image_tokens
                    input_ids_infer, _ = self.uni_prompting((conditions, image_tokens_infer), 't2i_gen')   # [1, 387]
                    if self.config.guidance_scale > 0:
                        uncond_input_ids, _ = self.uni_prompting(([''] * batch_size_t2i, image_tokens_infer), 't2i_gen')
                    # [1, 387], == [PAD] * 126 + <|t2i|> + <|endoftext|> + <|endoftext|> + <|soi|> + [MASK] * 256 + <|eoi|> ## no prompt
                        attention_mask1 = create_attention_mask_predict_next(torch.cat([input_ids_infer, uncond_input_ids], dim=0),    # [2, 387]
                                                                            pad_id=int(self.uni_prompting.sptids_dict['<|pad|>']),
                                                                            soi_id=int(self.uni_prompting.sptids_dict['<|soi|>']),
                                                                            eoi_id=int(self.uni_prompting.sptids_dict['<|eoi|>']),
                                                                            rm_pad_in_image=True)
                    else:
                        attention_mask1 = create_attention_mask_predict_next(input_ids_infer,
                                                                            pad_id=int(self.uni_prompting.sptids_dict['<|pad|>']),
                                                                            soi_id=int(self.uni_prompting.sptids_dict['<|soi|>']),
                                                                            eoi_id=int(self.uni_prompting.sptids_dict['<|eoi|>']),
                                                                            rm_pad_in_image=True)
                        uncond_input_ids = None
                    if self.config.get("mask_schedule", None) is not None:
                        schedule = self.config.mask_schedule.schedule
                        arg = self.config.mask_schedule.get("params", {})
                        mask_schedule = get_mask_chedule(schedule, **arg)
                    else:
                        mask_schedule = get_mask_chedule(self.config.training.get("mask_schedule", "cosine"))
                    

                    # input_ids_infer=input_ids_infer.repeat_interleave(self.num_generations,dim=0)
                    # attention_mask1=attention_mask1.repeat_interleave(self.num_generations,dim=0)
                    # uncond_input_ids=uncond_input_ids.repeat_interleave(self.num_generations,dim=0)

                    
                    # with torch.no_grad():
                    gen_token_ids,logits = self.model.t2i_generate(
                        input_ids=input_ids_infer,
                        uncond_input_ids=uncond_input_ids,
                        attention_mask=attention_mask1,
                        guidance_scale=self.config.guidance_scale,
                        temperature=self.config.training.get("generation_temperature",1e-5),
                        timesteps=self.config.generation_timesteps,
                        noise_schedule=mask_schedule,
                        noise_type=self.config.training.get("noise_type", "mask"),
                        seq_len=self.config.model.showo.num_vq_tokens,
                        uni_prompting=self.uni_prompting,
                        config=self.config,
                        return_logits=True,
                    )
                    local_gen_token_ids = gen_token_ids#[self.local_rank::self.world_size]
                    local_logits = logits#[self.local_rank::self.world_size]
                    global_id.append(local_gen_token_ids)
                    global_logits.append(local_logits)

                    del input_ids_infer, attention_mask1, uncond_input_ids
                    del gen_token_ids, logits
                    torch.cuda.empty_cache()
                gen_token_ids=torch.cat(global_id)
                logits=torch.cat(global_logits)
                per_token_logps=torch.log(logits)
                del global_id,global_logits,logits
                torch.cuda.empty_cache()
                gen_token_ids = torch.clamp(gen_token_ids, max=self.config.model.showo.codebook_size - 1, min=0)
                images = self.vq_model.decode_code(gen_token_ids)
                images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
                images *= 255.0
                images = images.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
                pil_images = [Image.fromarray(image) for image in images]
                del gen_token_ids
                for j in range(len(pil_images)):
                    gen_image = pil_images[j]
                    gen_image.save(os.path.join(save_dir, f"part_{self.world_size*j+self.local_rank}.png"))
                    # gen_image.save(os.path.join('/home/daigaole/code/ex/showo_feat/ref_image/adrien_brody',f"{counter}.png"))
                    # counter+=1
                    del gen_image
                torch.cuda.empty_cache()



                # if isinstance(images, np.ndarray):
                #     images = torch.from_numpy(images).to(self.args.device)
                # if not images.is_contiguous():
                #     images = images.contiguous()
                # all_images_list = [torch.zeros_like(images) for _ in range(self.world_size)]
                # dist.all_gather(all_images_list, images)
                # print('all',torch.cat(all_images_list, dim=0).shape)
                # all_images = torch.cat(all_images_list, dim=0)

                # all_logits_list = [torch.zeros_like(logits) for _ in range(self.world_size)]
                # dist.all_gather(all_logits_list, logits)
                # all_logits = torch.cat(all_logits_list, dim=0)

                # all_images = torch.clamp((all_images + 1.0) / 2.0, min=0.0, max=1.0)
                # all_images *= 255.0
                # all_images = all_images.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
                # print('all_numpy',all_images.shape)
                # if self.local_rank == 0:
                #     pil_images = [Image.fromarray(image) for image in all_images]
                #     print('len',len(pil_images))
                #     # tmp=counter
                #     for j in range(len(pil_images)):
                #         gen_image = pil_images[j]
                #         gen_image.save(os.path.join(save_dir, f"part_{j}.png"))
                #         # gen_image.save(os.path.join('/home/daigaole/code/ex/showo_feat/ref_image/adrien_brody',f"{counter}.png"))
                #         # counter+=1
                #         del gen_image
                #     # counter=tmp
                #     # for l in global_logits:
                #     #     torch.save(l,os.path.join('/home/daigaole/code/ex/showo_feat/ref_image/adrien_brody',f"{counter}.pt"))
                #     #     counter+=1
                #     del pil_images, images
                #     torch.cuda.empty_cache()

                self.model.config.mask_token_id=mti
                # logits=all_logits
                # del all_logits,all_images
                # continue

                # #test token:
                # ids=gen_token_ids[0]
                # ids1=gen_token_ids[1]
                # image_path='/home/daigaole/code/ex/showo_feat/tmp_result/0_ref.png'
                # image_ori = Image.open(image_path).convert("RGB")

                # image = image_transform(image_ori, resolution = self.config.dataset.params.resolution).to(self.args.device)
                # image = image.unsqueeze(0)
                # image_tokens_mmu = self.vq_model.get_code(image)
                # image_tokens = image_tokens_mmu + len(self.uni_prompting.text_tokenizer)

                # print('ids_0',ids,ids.shape)
                # print('ids_1',ids1,ids1.shape)
                # print('input ids',image_tokens,image_tokens.shape)
                # #understand test
                # top_k=1
                # question='please score the handsomeness of the person in the image using a number ranging from 0 to 10 and output the reasons.\n'
                # input_ids = self.uni_prompting.text_tokenizer(['USER: ' + question + ' ASSISTANT:'])['input_ids']
                # input_ids = torch.tensor(input_ids).to(self.args.device)

                # input_ids = torch.cat([
                #     (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|mmu|>']).to(self.args.device),
                #     (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|soi|>']).to(self.args.device),
                #     image_tokens,
                #     (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|eoi|>']).to(self.args.device),
                #     (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|sot|>']).to(self.args.device),
                #     input_ids
                # ], dim=1).long()
                # print('processed',input_ids,input_ids.shape)
                # attention_mask = create_attention_mask_for_mmu(input_ids.to(self.args.device),
                #                                                 eoi_id=int(self.uni_prompting.sptids_dict['<|eoi|>']))

                # cont_toks_list = self.ref_model.mmu_generate(input_ids, attention_mask=attention_mask,
                #                             max_new_tokens=100, top_k=top_k,
                #                             eot_token=self.uni_prompting.sptids_dict['<|eot|>'])

                # cont_toks_list = torch.stack(cont_toks_list).squeeze()[None]

                # text = self.uni_prompting.text_tokenizer.batch_decode(cont_toks_list, skip_special_tokens=True)[0].strip()
                # print(text)
                

                
                # load save logit
                load_dir='/home/daigaole/code/ex/showo_feat/ref_image/adrien_brody'
                save_logits=[]
                # for idx in range(100):
                #     l=torch.load(os.path.join(load_dir,f"{idx}.pt"))
                #     print(idx,l.max(),l.min())
                # return
                for j in range(self.group):
                    random_numbers = random.sample(range(100), 10)
                    tmp_list=[]
                    for idx in random_numbers:
                        l=torch.load(os.path.join(load_dir,f"{idx}.pt"))
                        l=torch.log(l)
                        tmp_list.append(l)
                    tmp_tensor=torch.stack(tmp_list).mean(dim=0)
                    save_logits.append(tmp_tensor)
                ref_per_token_logps=torch.cat(save_logits).to(self.args.device)
                del save_logits
                # if not signal:
                #     self.save_logits=logits
                #     signal=True
                # ref_per_token_logps=self.save_logits
                    
                
                #calculate the rewards
                # rewards=torch.zeros(self.num_generations*batch_size_t2i).to(self.args.device)
                reward_list=[]
                
                # question='Please output a score ranging from 0 to 10 to represent the correctness of the following question:\n'+'Is '
                # question='How much do you think that '
                # for token in new_tokens_stage_1:
                #     question+=token
                # for token in new_tokens_stage_2:
                #     question+=token
                # question+='<adrien_brody> in the image?\n'
                # question+='Please use a score ranging from 0 to 10 to represent.\n'
                # # question+='Only a score is needed,please don\'t output yes or no.\n'
                image_path='/home/daigaole/code/ex/showo_feat/tmp_result/'
                path_list=[os.path.join(image_path,f"part_{self.world_size*j+self.local_rank}.png") for j in range(self.group)]
                all_path_list=[os.path.join(image_path,f"part_{j}.png") for j in range(self.group*self.num_generations)]
                # for path in path_list:
                #     image_ori = Image.open(path).convert("RGB")
                #     image = image_transform(image_ori, resolution = self.config.dataset.params.resolution).to(self.args.device)
                #     image = image.unsqueeze(0)
                #     image_tokens_mmu = self.vq_model.get_code(image)
                #     image_tokens = image_tokens_mmu + len(self.uni_prompting.text_tokenizer)
                #     top_k=1
                    
                #     input_ids = self.uni_prompting.text_tokenizer(['USER: ' + question + ' ASSISTANT:'])['input_ids']
                #     input_ids = torch.tensor(input_ids).to(self.args.device)

                #     input_ids = torch.cat([
                #         (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|mmu|>']).to(self.args.device),
                #         (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|soi|>']).to(self.args.device),
                #         image_tokens,
                #         (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|eoi|>']).to(self.args.device),
                #         (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|sot|>']).to(self.args.device),
                #         input_ids
                #     ], dim=1).long()
                #     # print('processed',input_ids,input_ids.shape)
                #     attention_mask = create_attention_mask_for_mmu(input_ids.to(self.args.device),
                #                                                     eoi_id=int(self.uni_prompting.sptids_dict['<|eoi|>']))

                #     cont_toks_list = self.ref_model.mmu_generate(input_ids, attention_mask=attention_mask,
                #                                 max_new_tokens=100, top_k=top_k,
                #                                 eot_token=self.uni_prompting.sptids_dict['<|eot|>'])

                #     cont_toks_list = torch.stack(cont_toks_list).squeeze()[None]

                #     text = self.uni_prompting.text_tokenizer.batch_decode(cont_toks_list, skip_special_tokens=True)[0].strip()
                #     print(text)
                #     text=text.lower()
                #     if 'yes' in text:
                #         reward_list.append(1)
                #     elif 'no' in text:
                #         reward_list.append(0)
                #     else:
                #         reward_list.append(0.5)
                # rewards1=torch.tensor(reward_list).float().reshape(self.group*self.num_generations*batch_size_t2i).to(self.args.device)


                #clip reward
                reward2_list=[]
                clip_model=SHOWO_P_CLIPEvaluator("cuda:0")
                for path in path_list:
                    clip_model.save_dir=path
                    sim=clip_model.evaluate_concept('adrien_brody','',0)
                    print('similarity',path,sim)
                    if sim<0.5:
                        sim=0
                    else:
                        sim=2*(sim-0.5)
                    reward2_list.append(sim)
                # print('score',sum(reward2_list)/len(reward2_list))
                rewards2=torch.tensor(reward2_list).float().reshape(self.group).to(self.args.device)
                

                #gpt-4o reward
                # try:
                #     ref_path = '/home/daigaole/code/ex/dataset/unictokens_data/concept/train/adrien_brody/0.png'
                #     img_path = [ref_path]
                #     # 拼接所有图像路径（all_path_list是所有图像的地址集合）
                #     for path in path_list:
                #         img_path.append(path)
                    
                #     # 调用GPT评分（仅local_rank=1执行）
                #     answer = glm_evaluate(img_path)
                #     answer = extract_and_clean_list(answer)
                #     print(f'gpt score:', answer)
                    
                #     # 转换为tensor并移到对应设备（与其他进程保持设备一致）
                #     reward3_list = [float(i/100) for i in answer]
                #     mean_re3=sum(reward3_list)/len(reward3_list)
                #     diff=mean_re3-0.8
                #     reward3_list=[i-diff for i in reward3_list]
                #     rewards3 = torch.tensor(reward3_list, dtype=torch.float32, device=self.args.device)
                # except:
                #     rewards3 = torch.tensor([0.8 for _ in range(self.group)], dtype=torch.float32, device=self.args.device)
                # rewards=rewards2*0.2+rewards3*0.8
                signal=1
                try:
                    # 1. 仅在local_rank=1时调用GPT评分（避免所有进程重复调用）
                    if self.local_rank == 1:
                        ref_path = '/home/daigaole/code/ex/dataset/unictokens_data/concept/train/adrien_brody/0.png'
                        img_path = [ref_path]
                        # 拼接所有图像路径（all_path_list是所有图像的地址集合）
                        for path in all_path_list:
                            img_path.append(path)
                        for _ in range(5):
                            # 调用GPT评分（仅local_rank=1执行）
                            answer = glm_evaluate(img_path)
                            answer = extract_and_clean_list(answer)
                            if len(answer)==len(all_path_list):
                                break
                        # if len(answer)>len(all_path_list):
                        #     answer=answer[:-len(all_path_list)]
                        logging.info(f'[local_rank={self.local_rank}] gpt score: {answer}')
                        
                        # 转换为tensor并移到对应设备（与其他进程保持设备一致）
                        reward3_list = [float(i/100) for i in answer]
                        mean_re3=sum(reward3_list)/len(reward3_list)
                        diff=mean_re3-0.8
                        reward3_list=[i-diff for i in reward3_list]
                        rewards3 = torch.tensor(reward3_list, dtype=torch.float32, device=self.args.device)
                        signal=0
                    else:
                        # 其他进程初始化空tensor（用于接收广播结果）
                        # 注意：需提前知道rewards3的形状，假设长度为self.group
                        rewards3 = torch.tensor([0.8 for _ in range(self.group*self.num_generations)], dtype=torch.float32, device=self.args.device)

                except Exception as e:
                    print(f'[local_rank={self.local_rank}] 评分计算失败，使用默认rewards2: {e}')
                signal_tensor = torch.tensor([signal], dtype=torch.int, device=self.args.device)
                dist.broadcast(signal_tensor, src=1)  # 同步 signal 到所有进程
                signal = signal_tensor.item()  # 所有进程的 signal 现在一致
                if signal==0:
                    # 2. 将local_rank=1的rewards3广播到所有进程
                    # 广播源为local_rank=1（确保该进程已计算出rewards3）
                    dist.broadcast(rewards3, src=1)  # src=1表示从local_rank=1广播到所有进程

                    # 3. 每个进程提取自己的reward（按local_rank索引，或按进程分片）
                    # 假设总进程数=group_size，每个进程取对应位置的元素
                    # 例如：总长度为self.group，每个进程取self.group // world_size个元素
                    world_size = dist.get_world_size()
                    total_length = self.group * self.num_generations

                    # 计算当前进程应处理的索引（交叉分片）
                    # 例如：总长度6，3进程时，进程0取0、3；进程1取1、4；进程2取2、5
                    indices = [i for i in range(total_length) if i % world_size == self.local_rank]

                    # 提取当前进程的reward
                    my_rewards3 = rewards3[indices].reshape(-1)

                    # 4. 与rewards2合成（当前进程仅用自己的reward片段）
                    rewards = rewards2 * 0.2 + my_rewards3 * 0.8
                else:
                    rewards=rewards2
                # reward_qa=torch.tensor(reward_qa_list).float().reshape(self.group).to(self.args.device)
                # reward_vqa=torch.tensor(reward_vqa_list).float().reshape(self.group).to(self.args.device)
                reward_text=torch.tensor(reward_text).float().reshape(self.group).to(self.args.device)
                rewards=rewards*0.7+reward_text*0.3
                counter+=1
                #update_best_results(path_list,rewards.cpu().detach().numpy().tolist(),counter)

                
                all_rewards_list = [torch.zeros_like(rewards) for _ in range(self.world_size)]
                dist.all_gather(all_rewards_list, rewards)
                all_rewards = torch.cat(all_rewards_list, dim=0)
                rewards=all_rewards
                if self.local_rank==0:
                    for r in range(len(rewards)):
                        logging.info(f"img {r}, reward {rewards[r]}")

                # if self.local_rank==0:
                #     print('max',logits.max())
                #     print('min',logits.min())
                # per_token_logps=torch.log(logits)
                # if self.local_rank==0:
                #     print(per_token_logps.requires_grad)
                # print(rewards,per_token_logps,ref_per_token_logps)


                #grpo
                mean_grouped_rewards = rewards.view(-1, self.num_generations*self.group).mean(dim=1)
                std_grouped_rewards = rewards.view(-1, self.num_generations*self.group).std(dim=1)

                # Normalize the rewards to compute the advantages
                mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations*self.group, dim=0)
                std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations*self.group, dim=0)
                advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
                dist.broadcast(advantages, src=0)
                advantages = advantages[self.local_rank::self.world_size]
                # print(advantages)

                self.model.train()
                # torch.set_grad_enabled(True)
                # per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1).unsqueeze(2)
                # if self.local_rank==0:
                #     print('bfmax',per_token_logps.max(),ref_per_token_logps.max())
                #     print('bfmin',per_token_logps.min(),ref_per_token_logps.min())
                # per_token_logps=normalize_logits(per_token_logps)
                # ref_per_token_logps=normalize_logits(ref_per_token_logps)
                if self.local_rank==0:
                    print('afmax',per_token_logps.max(),ref_per_token_logps.max())
                    print('afmin',per_token_logps.min(),ref_per_token_logps.min())
                log_ratio =per_token_logps - ref_per_token_logps
                log_ratio = torch.clamp(log_ratio, -10, 10)
                ratio = torch.exp(log_ratio)

                
                per_token_loss= ratio* advantages.unsqueeze(1).unsqueeze(2)
                if self.local_rank==0:
                    print('bf mean of per_token_loss',per_token_loss.mean())
                    print('ratio',ratio)
                per_token_kl = ratio - log_ratio - 1

                

                per_token_loss = -(per_token_loss - self.beta * per_token_kl)

                if self.local_rank==0:
                    logging.info(f"mean of kl: {per_token_kl.mean()}")
                    logging.info(f"mean of per_token_loss: {per_token_loss.mean()}")
                # print("completion_mask shape:", completion_mask.shape)
                completion_mask=torch.ones_like(per_token_loss)
                loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

                #log
                if self.deepspeed_enabled:
                    self.model.backward(loss)
                    self.model.step()
                else:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                # if self.args.t2i_data and (epoch+1) % 10 == 0:
                #     print('epoch',epoch+1,'loss',loss.item())

                loss_tensor = torch.tensor([loss.item()], device=loss.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                world_size = dist.get_world_size()
                avg_loss = loss_tensor.item() / world_size
                if dist.get_rank() == 0:
                    loss_list.append(avg_loss)
                    logging.info(f"{batch_idx} loss: {avg_loss:.4f}")

                # for (n1, p1), (n2, p2) in zip(self.model.named_parameters(), self.ref_model.named_parameters()):
                #     if torch.allclose(p1, p2)!=True:
                #         print(f"parameter {n1} is not equal")
                import gc
                del rewards, per_token_logps, ref_per_token_logps
                gc.collect()
                torch.cuda.empty_cache()
            if epoch%3==0 and epoch>0:
                save_distributed_model(self.model,self.optimizer,'/home/daigaole/code/ex/showo_feat/result',epoch=epoch)
