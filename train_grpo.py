import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
from PIL import Image
from pdata import image_transform
import numpy as np
import torch
import torch.utils.data
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
from gpttest import chat_with_images_gpt
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
                config_params='/mnt/public/gpfs-jd/data/lh/ex/showo_feat/ds_config.json',
                model_parameters=filter(lambda p: p.requires_grad, model.parameters())
            )
            self.model.module.gradient_checkpointing_enable()
        else:
            self.model=model
            self.optimizer=optimizer
        self.num_generations = args.num_gen  # = G in the GRPO paper
        self.group=int(self.num_generations/4)
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
                for group_id in range(self.group):
                    mti=self.model.config.mask_token_id
                    self.model.config.mask_token_id = self.model.showo.get_input_embeddings().num_embeddings - 1
                    mask_token_id = self.model.showo.get_input_embeddings().num_embeddings - 1
                    image_tokens_infer = torch.ones((batch_size_t2i, self.config.model.showo.num_vq_tokens),
                                            dtype=torch.long, device=self.args.device) * mask_token_id
                    
                    
                    save_dir='/mnt/public/gpfs-jd/data/lh/ex/showo_feat/tmp_result/'
                    condition='A photo of '
                    for token in new_tokens_stage_1:
                        condition+=token
                    for token in new_tokens_stage_2:
                        condition+=token
                    condition+='<adrien_brody>.\n'
                    conditions = [condition] * batch_size_t2i
                    
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
                    print(self.num_generations,self.local_rank,group_id)

                    # input_ids_infer=input_ids_infer.repeat_interleave(self.num_generations,dim=0)
                    # attention_mask1=attention_mask1.repeat_interleave(self.num_generations,dim=0)
                    # uncond_input_ids=uncond_input_ids.repeat_interleave(self.num_generations,dim=0)

                    # print('model-basemodel')
                    # for (n1, p1), (n2, p2) in zip(self.model.named_parameters(), self.basemodel.named_parameters()):
                    #     if torch.allclose(p1, p2)!=True:
                    #         print(f"parameter {n1} is not equal")
                    # print('refmodel-basemodel')
                    # for (n1, p1), (n2, p2) in zip(self.ref_model.named_parameters(), self.basemodel.named_parameters()):
                    #     if torch.allclose(p1, p2)!=True:
                    #         print(f"parameter {n1} is not equal")
                    
                    # with torch.no_grad():
                    gen_token_ids,logits = self.model.t2i_generate(
                        input_ids=input_ids_infer,
                        uncond_input_ids=uncond_input_ids,
                        attention_mask=attention_mask1,
                        guidance_scale=self.config.guidance_scale,
                        temperature=self.config.training.get("generation_temperature", 1.0),
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
                del global_id,global_logits
                torch.cuda.empty_cache()
                gen_token_ids = torch.clamp(gen_token_ids, max=self.config.model.showo.codebook_size - 1, min=0)
                images = self.vq_model.decode_code(gen_token_ids)
                images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
                images *= 255.0
                images = images.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
                pil_images = [Image.fromarray(image) for image in images]
                for j in range(len(pil_images)):
                    gen_image = pil_images[j]
                    gen_image.save(os.path.join(save_dir, f"part_{self.world_size*j+self.local_rank}.png"))
                    # gen_image.save(os.path.join('/mnt/public/gpfs-jd/data/lh/ex/showo_feat/ref_image/adrien_brody',f"{counter}.png"))
                    # counter+=1
                    del gen_image



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
                #         # gen_image.save(os.path.join('/mnt/public/gpfs-jd/data/lh/ex/showo_feat/ref_image/adrien_brody',f"{counter}.png"))
                #         # counter+=1
                #         del gen_image
                #     # counter=tmp
                #     # for l in global_logits:
                #     #     torch.save(l,os.path.join('/mnt/public/gpfs-jd/data/lh/ex/showo_feat/ref_image/adrien_brody',f"{counter}.pt"))
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
                # image_path='/mnt/public/gpfs-jd/data/lh/ex/showo_feat/tmp_result/0_ref.png'
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
                load_dir='/mnt/public/gpfs-jd/data/lh/ex/showo_feat/ref_image/adrien_brody'
                save_logits=[]
                for j in range(self.group):
                    random_numbers = random.sample(range(100), 30)
                    tmp_list=[]
                    for idx in random_numbers:
                        l=torch.load(os.path.join(load_dir,f"{idx}.pt"))
                        # l=torch.softmax(l,dim=-1)
                        tmp_list.append(l)
                    tmp_tensor=torch.stack(tmp_list).mean(dim=0)
                    save_logits.append(tmp_tensor)
                ref_per_token_logps=torch.cat(save_logits).to(self.args.device)
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
                image_path='/mnt/public/gpfs-jd/data/lh/ex/showo_feat/tmp_result/'
                path_list=[os.path.join(image_path,f"part_{self.world_size*j+self.local_rank}.png") for j in range(self.group)]
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
                    reward2_list.append(sim)
                # print('score',sum(reward2_list)/len(reward2_list))
                rewards2=torch.tensor(reward2_list).float().reshape(self.group).to(self.args.device)

                #gpt-4o reward
                # reward3_list=[]
                # prompt='How much do you think the man in the first image appears in the second image?\nPlease use a number ranging from 0 to 1 to represent.\nPlease only output a number.\n'
                # ref_path='/mnt/public/gpfs-jd/data/lh/ex/dataset/unictokens_data/concept/train/adrien_brody/0.png'
                # for path in path_list:
                #     answer=chat_with_images_gpt(prompt,[ref_path,path])
                #     answer=extract_single_number(answer)
                #     print(path,'gpt score:',answer)
                #     reward3_list.append(answer)
                # rewards3=torch.tensor(reward3_list).float().reshape(self.num_generations*batch_size_t2i*self.group).to(self.args.device)

                # rewards=rewards2*0.2+rewards3*0.8
                
                rewards=rewards2



                
                all_rewards_list = [torch.zeros_like(rewards) for _ in range(self.world_size)]
                dist.all_gather(all_rewards_list, rewards)
                all_rewards = torch.cat(all_rewards_list, dim=0)
                rewards=all_rewards

                
                per_token_logps=logits
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
                # print(advantages)
                advantages = advantages[self.local_rank::self.world_size]
                # print(advantages)

                self.model.train()
                # torch.set_grad_enabled(True)
                # per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1).unsqueeze(2)

                per_token_logps=normalize_logits(per_token_logps)
                ref_per_token_logps=normalize_logits(ref_per_token_logps)
                log_ratio =per_token_logps - ref_per_token_logps
                ratio = torch.exp(log_ratio)
                
                
                per_token_loss= ratio* advantages.unsqueeze(1).unsqueeze(2)

                per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

                if self.local_rank==0:
                    print('mean of kl',per_token_kl.mean())
                    print('mean of per_token_loss',per_token_loss.mean())

                per_token_loss = -(per_token_loss - self.beta * per_token_kl)

                # print("completion_mask shape:", completion_mask.shape)
                completion_mask=torch.ones_like(per_token_loss)
                loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

                #log
                if self.deepspeed_enabled:
                    self.model.backward(loss)
                    self.model.step()
                else:
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
                    print(f"loss: {avg_loss:.4f}")

                # for (n1, p1), (n2, p2) in zip(self.model.named_parameters(), self.ref_model.named_parameters()):
                #     if torch.allclose(p1, p2)!=True:
                #         print(f"parameter {n1} is not equal")
                import gc
                del gen_token_ids, logits, rewards, per_token_logps, ref_per_token_logps
                gc.collect()
                torch.cuda.empty_cache()
