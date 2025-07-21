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
from datasets import Dataset, IterableDataset
from packaging import version
from tqdm import tqdm
from clip_eval import SHOWO_P_CLIPEvaluator
from models import Showo, MAGVITv2, get_mask_chedule
from clip.model import build_model
from training.prompting_utils import UniversalPrompting, create_attention_mask_predict_next, create_attention_mask_for_mmu
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
from transformers.utils import is_peft_available
from training.utils import get_config, flatten_omega_conf, mask_or_random_replace_tokens, AverageMeter
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url
import shutil

import copy
import open_clip
from open_clip import create_model_and_transforms
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
class unic_grpo(Trainer):
    def __init__(
        self,
        model,
        reward_funcs,
        args,
        train_args,
        config,
        dataset,
        vq_model,
        uni_prompting,
        optimizer,
        tokenizer,
        peft_config: Optional["PeftConfig"] = None,
        attn_implementation: str = "flash_attention_2",
    ):

        # freeze all vision encoders
        # for name, param in model.named_parameters():
        #     if name.startswith("vision_model") or name.startswith("aligner") or name.startswith("gen"): # choose whatever you like here
        #         param.requires_grad = False



        # print(dir(model))
        # model.language_model.config._attn_implementation == "flash_attention_2"
        # model.language_model.config.use_cache = False
        # model.language_model.gradient_checkpointing_enable()

        # # Reference model
        self.ref_model=clone_model(model)
        self.ref_model.eval()
        for name, param in self.ref_model.named_parameters():
            param.requires_grad = False

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


        self.num_generations = args.num_gen  # = G in the GRPO paper
        self.beta = train_args.beta
        self._metrics = defaultdict(list)
        self.model_accepts_loss_kwargs = False
        self.dataset=dataset
        self.args=args
        self.model=model
        self.optimizer=optimizer
        self.config=config
        self.vq_model=vq_model
        self.uni_prompting=uni_prompting
        self.tokenizer=tokenizer
        self.beta=0.01
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

    def train(self, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # other setting
        self.model.config.mask_token_id = self.model.showo.get_input_embeddings().num_embeddings - 1
        self.model.mask_token_id = self.model.showo.get_input_embeddings().num_embeddings - 1
        self.ref_model.config.mask_token_id = self.model.showo.get_input_embeddings().num_embeddings - 1
        self.ref_model.mask_token_id = self.model.showo.get_input_embeddings().num_embeddings - 1
        mask_schedule = get_mask_chedule(self.config.training.get("mask_schedule", "cosine"))
        mask_id = self.model.mask_token_id
        mask_dtype = self.model.showo.get_input_embeddings().weight.dtype
        self.model.output_size = self.config.new_total_vocab
        self.ref_model.output_size = self.config.new_total_vocab
        for epoch in range(self.args.epoch):
            print(f"Epoch {epoch+1}")
            loss_list = []
            if self.args.t2i_data:
                loss_t2i_list = []
            if self.args.mmu_data:
                loss_mmu_list = []
            for batch, batch_idx, dataloader_idx in tqdm(self.dataset):
                if self.args.t2i_data:
                    
                    batch_size_t2i = batch["t2i_flow"]["images"].shape[0]
                    pixel_values, texts = batch["t2i_flow"]["images"], batch["t2i_flow"]["conditions"]
                    pixel_values = pixel_values.to(self.args.device)
                    input_ids_t2i, labels_t2i, mask_prob, image_tokens_ori = prepare_inputs_and_labels(mask_id,
                                                                                            self.config,
                                                                                            self.vq_model,
                                                                                            self.uni_prompting,
                                                                                            mask_schedule,
                                                                                            pixel_values,
                                                                                            texts,
                                                                                            is_train=True,)
                    # print('input',input_ids_t2i.shape)
                    attention_mask_t2i = create_attention_mask_predict_next(input_ids_t2i,
                                                                        pad_id=int(self.uni_prompting.sptids_dict['<|pad|>']),
                                                                        soi_id=int(self.uni_prompting.sptids_dict['<|soi|>']),
                                                                        eoi_id=int(self.uni_prompting.sptids_dict['<|eoi|>']),
                                                                        rm_pad_in_image=True,
                                                                        return_inverse_mask=True)
                    attention_mask_t2i = attention_mask_t2i.to(mask_dtype)
                    # print('nask',attention_mask_t2i.shape)
                if self.args.mmu_data:
                    batch_size_mmu = batch["mmu_flow"]["images"].shape[0]
                    pixel_values_mmu, input_ids_mmu, labels_mmu = (batch["mmu_flow"]["images"],
                                                                batch["mmu_flow"]["input_ids"],
                                                                batch["mmu_flow"]["labels"])
                    pixel_values_mmu = pixel_values_mmu.to(self.args.device, non_blocking=True)
                    input_ids_mmu = input_ids_mmu.to(self.args.device, non_blocking=True)
                    image_tokens_mmu = self.vq_model.get_code(pixel_values_mmu)
                    image_tokens_mmu = image_tokens_mmu + len(self.uni_prompting.text_tokenizer)
                    
                    input_ids_mmu = torch.cat([
                                (torch.ones(input_ids_mmu.shape[0], 1) * self.uni_prompting.sptids_dict['<|mmu|>']).to(self.args.device),
                                (torch.ones(input_ids_mmu.shape[0], 1) * self.uni_prompting.sptids_dict['<|soi|>']).to(self.args.device),
                                image_tokens_mmu,
                                (torch.ones(input_ids_mmu.shape[0], 1) * self.uni_prompting.sptids_dict['<|eoi|>']).to(self.args.device),
                                input_ids_mmu,
                            ], dim=1).long()

                    labels_mmu = torch.cat([
                                (torch.ones(input_ids_mmu.shape[0], 1) * self.uni_prompting.ignore_id).to(self.args.device),
                                (torch.ones(input_ids_mmu.shape[0], 1) * self.uni_prompting.ignore_id).to(self.args.device),
                                torch.ones_like(image_tokens_mmu) * self.uni_prompting.ignore_id,
                                (torch.ones(input_ids_mmu.shape[0], 1) * self.uni_prompting.ignore_id).to(self.args.device),
                                labels_mmu.to(self.args.device)
                            ], dim=1).long()
                    
                    
                    attention_mask_mmu = create_attention_mask_for_mmu(input_ids_mmu.to(self.args.device),
                                                                        eoi_id=int(self.uni_prompting.sptids_dict['<|eoi|>']))
                    attention_mask_mmu = attention_mask_mmu.to(mask_dtype)

                if self.args.t2i_data and self.args.mmu_data:
                    attention_mask = torch.cat([attention_mask_t2i, attention_mask_mmu], dim=0)
                    input_ids = torch.cat([input_ids_t2i, input_ids_mmu], dim=0)
                    labels = torch.cat([labels_t2i, labels_mmu], dim=0)
                elif self.args.t2i_data:
                    attention_mask = attention_mask_t2i
                    input_ids = input_ids_t2i
                    labels = labels_t2i
                    batch_size_mmu = 0
                elif self.args.mmu_data:
                    attention_mask = attention_mask_mmu
                    input_ids = input_ids_mmu
                    labels = labels_mmu
                    batch_size_t2i = 0
                else:
                    raise ValueError("No dataset loaded")
                self.optimizer.zero_grad()
                # input_ids=input_ids.repeat_interleave(self.num_generations,dim=0)
                # attention_mask=attention_mask.repeat_interleave(self.num_generations,dim=0)
                # labels=labels.repeat_interleave(self.num_generations,dim=0)
                # logits, loss_t2i, loss_lm, loss_mmu = self.model(
                #     input_ids=input_ids,
                #     input_embeddings=None,
                #     attention_mask=attention_mask,
                #     labels=labels,
                #     label_smoothing=0.0,
                #     batch_size_t2i=batch_size_t2i,
                #     batch_size_lm=0,
                #     batch_size_mmu=batch_size_mmu,
                #     max_seq_length=128,
                # )
                # image_logits = logits[:self.num_generations*batch_size_t2i, -self.config.model.showo.num_vq_tokens:, -self.config.model.showo.codebook_size:]
                # assert image_logits.shape == (self.num_generations*batch_size_t2i, self.config.model.showo.num_vq_tokens, self.config.model.showo.codebook_size)
                # # 得到通过最大值位置得到分类 id [batch_size_t2i, config.model.showo.num_vq_tokens]
                # image_gen_ids = torch.argmax(image_logits, dim=-1) 
                # assert image_gen_ids.shape == (self.num_generations*batch_size_t2i, self.config.model.showo.num_vq_tokens)
                # completion_ids = torch.clamp(image_gen_ids, max=self.config.model.showo.codebook_size - 1, min=0)
                # completion_mask=torch.ones_like(completion_ids)
                # torch.set_grad_enabled(True)
                # prompt_all_ids = torch.cat([input_ids, completion_ids], dim=1)
                # embed_layer = self.model.showo.get_input_embeddings()
                # input_embeds=embed_layer(prompt_all_ids)
                # print(attention_mask.shape,completion_mask.shape)
                # input_mask= (~input_ids==int(self.uni_prompting.sptids_dict['<|pad|>'])).int()
                # attention_mask = torch.cat([input_mask, completion_mask], dim=1)
                #log_p
                # per_token_logps = image_logits
                # print(per_token_logps)
                # with torch.inference_mode():
                #     if self.ref_model is not None:
                #         ref_logits, ref_loss_t2i, ref_loss_lm, ref_loss_mmu = self.ref_model(
                #             input_ids=input_ids,
                #             input_embeddings=None,
                #             attention_mask=attention_mask,
                #             labels=labels,
                #             label_smoothing=0.0,
                #             batch_size_t2i=batch_size_t2i,
                #             batch_size_lm=0,
                #             batch_size_mmu=batch_size_mmu,
                #             max_seq_length=128,
                #         )
                #         ref_image_logits = ref_logits[:self.num_generations*batch_size_t2i, -self.config.model.showo.num_vq_tokens:, -self.config.model.showo.codebook_size:]
                #         assert ref_image_logits.shape == (self.num_generations*batch_size_t2i, self.config.model.showo.num_vq_tokens, self.config.model.showo.codebook_size)
                #         # 得到通过最大值位置得到分类 id [batch_size_t2i, config.model.showo.num_vq_tokens]
                #         ref_per_token_logps = ref_image_logits

                #     else:
                #         # dummy ref_per_token_logps
                #         ref_per_token_logps = torch.zeros_like(per_token_logps)
                # # torch.set_grad_enabled(False)




                #realize t2i inference
                # m=self.config.mode
                # bs=self.config.batch_size

                # gs=self.config.guidance_scale
                # lvs=self.config.model.showo.llm_vocab_size

                # self.config.mode = 't2i'
                # self.config.batch_size = 2
                self.config.model.showo.llm_vocab_size = len(self.tokenizer) - 10
                self.config.generation_timesteps = 50
                self.config.guidance_scale = 5

                nums_new_token_i_stage_1 = self.args.nums_new_token_i_stage_1
                nums_new_token_i_stage_2 = self.args.nums_new_token_i_stage_2
                new_tokens_stage_1 = [f"<token_{i}>" for i in range(nums_new_token_i_stage_1)]
                new_tokens_stage_2 = [f"<token_{i}>" for i in range(nums_new_token_i_stage_1, nums_new_token_i_stage_1 + nums_new_token_i_stage_2)]

                mti_ref=self.ref_model.config.mask_token_id
                self.ref_model.config.mask_token_id = self.ref_model.showo.get_input_embeddings().num_embeddings - 1
                mask_token_id = self.ref_model.showo.get_input_embeddings().num_embeddings - 1
                image_tokens_infer = torch.ones((batch_size_t2i, self.config.model.showo.num_vq_tokens),
                                        dtype=torch.long, device=self.args.device) * mask_token_id
                
                mti=self.model.config.mask_token_id
                self.model.config.mask_token_id = self.model.showo.get_input_embeddings().num_embeddings - 1
                mask_token_id = self.model.showo.get_input_embeddings().num_embeddings - 1
                image_tokens_infer = torch.ones((batch_size_t2i, self.config.model.showo.num_vq_tokens),
                                        dtype=torch.long, device=self.args.device) * mask_token_id
                
                
                
                
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
                    self.args = self.config.mask_schedule.get("params", {})
                    mask_schedule = get_mask_chedule(schedule, **args)
                else:
                    mask_schedule = get_mask_chedule(self.config.training.get("mask_schedule", "cosine"))
                
                input_ids_infer=input_ids_infer.repeat_interleave(self.num_generations,dim=0)
                attention_mask1=attention_mask1.repeat_interleave(self.num_generations,dim=0)
                uncond_input_ids=uncond_input_ids.repeat_interleave(self.num_generations,dim=0)

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


                with torch.no_grad():
                    gen_token_ids_ref,logits_ref = self.ref_model.t2i_generate(
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

                save_dir='/home/daigaole/code/ex/showo_feat/tmp_result'
                gen_token_ids = torch.clamp(gen_token_ids, max=self.config.model.showo.codebook_size - 1, min=0)
                images = self.vq_model.decode_code(gen_token_ids)
                images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
                images *= 255.0
                images = images.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
                pil_images = [Image.fromarray(image) for image in images]
                
                for j in range(len(pil_images)):
                    gen_image = pil_images[j]
                    gen_image.save(os.path.join(save_dir, f"{j}.png"))
                
                gen_token_ids = torch.clamp(gen_token_ids_ref, max=self.config.model.showo.codebook_size - 1, min=0)
                images = self.vq_model.decode_code(gen_token_ids)
                images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0)
                images *= 255.0
                images = images.permute(0, 2, 3, 1).detach().cpu().numpy().astype(np.uint8)
                pil_images = [Image.fromarray(image) for image in images]

                for j in range(batch_size_t2i*self.num_generations):
                    gen_image = pil_images[j]
                    gen_image.save(os.path.join(save_dir, f"{j}_ref.png")) 
                # print(batch_size_t2i,self.num_generations,len(pil_images))
                
            
                self.model.config.mask_token_id=mti
                self.ref_model.config.mask_token_id=mti_ref



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
                

                
                
                
                
                #calculate the rewards
                # rewards=torch.zeros(self.num_generations*batch_size_t2i).to(self.args.device)
                reward_list=[]
                
                question='Please output a score ranging from 0 to 10 to represent the correctness of the following question:\n'+'Is '
                question='How much do you think that '
                for token in new_tokens_stage_1:
                    question+=token
                for token in new_tokens_stage_2:
                    question+=token
                question+='<adrien_brody> in the image?\n'
                question+='Please use a score ranging from 0 to 10 to represent.\n'
                # question+='Only a score is needed,please don\'t output yes or no.\n'
                image_path='/home/daigaole/code/ex/showo_feat/tmp_result/'
                path_list=[os.path.join(image_path,f"{j}.png") for j in range(batch_size_t2i*self.num_generations)]
                for path in path_list:
                    image_ori = Image.open(path).convert("RGB")
                    image = image_transform(image_ori, resolution = self.config.dataset.params.resolution).to(self.args.device)
                    image = image.unsqueeze(0)
                    image_tokens_mmu = self.vq_model.get_code(image)
                    image_tokens = image_tokens_mmu + len(self.uni_prompting.text_tokenizer)
                    top_k=1
                    
                    input_ids = self.uni_prompting.text_tokenizer(['USER: ' + question + ' ASSISTANT:'])['input_ids']
                    input_ids = torch.tensor(input_ids).to(self.args.device)

                    input_ids = torch.cat([
                        (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|mmu|>']).to(self.args.device),
                        (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|soi|>']).to(self.args.device),
                        image_tokens,
                        (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|eoi|>']).to(self.args.device),
                        (torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|sot|>']).to(self.args.device),
                        input_ids
                    ], dim=1).long()
                    # print('processed',input_ids,input_ids.shape)
                    attention_mask = create_attention_mask_for_mmu(input_ids.to(self.args.device),
                                                                    eoi_id=int(self.uni_prompting.sptids_dict['<|eoi|>']))

                    cont_toks_list = self.ref_model.mmu_generate(input_ids, attention_mask=attention_mask,
                                                max_new_tokens=100, top_k=top_k,
                                                eot_token=self.uni_prompting.sptids_dict['<|eot|>'])

                    cont_toks_list = torch.stack(cont_toks_list).squeeze()[None]

                    text = self.uni_prompting.text_tokenizer.batch_decode(cont_toks_list, skip_special_tokens=True)[0].strip()
                    print(text)
                    text=text.lower()
                    if 'yes' in text:
                        reward_list.append(1)
                    elif 'no' in text:
                        reward_list.append(0)
                    else:
                        reward_list.append(0.5)
                rewards1=torch.tensor(reward_list).float().reshape(self.num_generations*batch_size_t2i).to(self.args.device)
                #clip reward
                reward2_list=[]
                
                # clip_model, _, _ = open_clip.create_model_and_transforms(
                #     model_name="ViT-B-32",
                #     pretrained="/home/daigaole/code/models/ViT-B-32/open_clip_pytorch_model.bin", 
                #     device=self.args.device,
                # )
                clip_model=SHOWO_P_CLIPEvaluator("cuda:0")
                for path in path_list:
                    clip_model.save_dir=path
                    sim=clip_model.evaluate_concept('adrien_brody','',0)
                    print(sim)
                    reward2_list.append(sim)
                print('score',sum(reward2_list)/len(reward2_list))
                rewards2=torch.tensor(reward2_list).float().reshape(self.num_generations*batch_size_t2i).to(self.args.device)
                rewards=rewards2
                per_token_logps=logits
                ref_per_token_logps=logits_ref
                rewards2=torch.tensor(reward2_list).float().reshape(self.num_generations*batch_size_t2i).to(self.args.device)
                rewards=rewards2
                per_token_logps=logits
                ref_per_token_logps=logits_ref



                #grpo
                mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
                std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

                # Normalize the rewards to compute the advantages
                mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
                std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
                advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

                torch.set_grad_enabled(True) 
                per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1).unsqueeze(2)
                # print(per_token_loss)
                per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
                per_token_loss = -(per_token_loss - self.beta * per_token_kl)
                # print("per_token_loss shape:", per_token_loss.shape)  # 实际形状
                # print("completion_mask shape:", completion_mask.shape)  # 应该是 (8, 1024)
                completion_mask = completion_mask.unsqueeze(-1)
                # print(per_token_loss,per_token_kl)
                loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

                #log
                loss.backward()
                self.optimizer.step()
                if self.args.t2i_data and (epoch+1) % 10 == 0:
                    print('epoch',epoch+1,'loss',loss.item())
                loss_list.append(loss.item())
                if self.args.t2i_data:
                    loss_t2i_list.append(loss_t2i.item())
                if self.args.mmu_data:
                    loss_mmu_list.append(loss_mmu.item())

            for (n1, p1), (n2, p2) in zip(self.model.named_parameters(), self.ref_model.named_parameters()):
                if torch.allclose(p1, p2):
                    print(f"parameter {n1} is equal")
