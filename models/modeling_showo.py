# coding=utf-8
# Copyright 2024 NUS Show Lab, HuggingFace.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F
from transformers import AutoConfig
from .modeling_utils import ConfigMixin, ModelMixin, register_to_config
from .sampling import cosine_schedule, mask_by_random_topk
from .phi import PhiForCausalLM

class Showo(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
            self,
            w_clip_vit,
            vocab_size,
            llm_vocab_size,
            llm_model_path='/home/daigaole/code/ex/phi-1_5',
            codebook_size=8192,
            num_vq_tokens=256,
            load_from_showo=True,
            **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.register_to_config(mask_token_id=vocab_size - 1)
        if load_from_showo:
            config = AutoConfig.from_pretrained(llm_model_path)
            self.showo = PhiForCausalLM(config)
        else:
            self.showo = PhiForCausalLM.from_pretrained(llm_model_path, attn_implementation='sdpa')
        print(f"Embeddings device: {self.showo.get_input_embeddings().weight.device}")
        print(f"Embeddings is meta: {self.showo.get_input_embeddings().weight.is_meta}")
        self.showo.resize_token_embeddings(self.vocab_size)
        self.output_size = self.vocab_size

        if self.w_clip_vit:
            self.mm_projector = torch.nn.Sequential(
                torch.nn.Linear(1024, 2048),
                torch.nn.GELU(),
                torch.nn.Linear(2048, 2048)
            )

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = True

    def forward(
            self,
            input_ids,
            input_embeddings=None,
            attention_mask=None,
            labels=None,
            label_smoothing=0.0,
            batch_size_t2i=0,
            batch_size_lm=0,
            batch_size_mmu=0,
            max_seq_length=128,
            labels_mask_text=None,
            labels_mask_image=None,
            **kwargs,
    ):

        if input_embeddings is None:
            logits = self.showo(input_ids=input_ids, attention_mask=attention_mask)['logits']
            # print(logits)
        else:
            logits = self.showo(inputs_embeds=input_embeddings, attention_mask=attention_mask)['logits']
        if labels is not None:
            # 1. Mask token prediction (discrete diffusion) for image generation
            # Note that, max_seq_length indicates the maximum number of text tokens, maybe a bit confused.
            loss_t2i = F.cross_entropy(
                logits[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1, self.output_size),
                labels[:batch_size_t2i, max_seq_length + 1:].contiguous().view(-1), ignore_index=-100,
            )

            # 2. Next token prediction for language modeling
            loss_lm = F.cross_entropy(
                logits[batch_size_t2i:batch_size_t2i + batch_size_lm, :-1].contiguous().view(-1, self.output_size),
                labels[batch_size_t2i:batch_size_t2i + batch_size_lm, 1:].contiguous().view(-1), ignore_index=-100,
            )

            # 3. Next token prediction for captioning/multimodal understanding
            loss_mmu = F.cross_entropy(
                logits[-batch_size_mmu:, :-1].contiguous().view(-1, self.output_size),
                labels[-batch_size_mmu:, 1:].contiguous().view(-1), ignore_index=-100,
            )

            return logits, loss_t2i, loss_lm, loss_mmu

        return logits

    def t2i_generate(
            self,
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            temperature=1.0,
            timesteps=18,  # ideal number of steps is 18 in maskgit paper
            guidance_scale=0,
            noise_schedule=cosine_schedule,
            generator: torch.Generator = None,
            config=None,
            return_logits=False,
            checkpoint=None,
            save_checkpoint=False,
            checkpoint_step=0,
            **kwargs,
    ):
        """
        Generate 1:1 similar to the original MaskGit repo
        https://github.com/google-research/maskgit/blob/main/maskgit/libml/parallel_decode.py#L79
        """
        if checkpoint is not None:
            step_start = checkpoint["step"]
            mask_token_id = checkpoint["mask_token_id"]
            num_vq_tokens = checkpoint["num_vq_tokens"]
            num_new_special_tokens = checkpoint["num_new_special_tokens"]
            input_ids = checkpoint["input_ids"]
            input_ids_minus_lm_vocab_size = checkpoint["input_ids_minus_lm_vocab_size"]
            uncond_prefix = checkpoint["uncond_prefix"]
            # 恢复温度参数状态（保证与第9步时一致）
            temperature = checkpoint["temperature"]
        else:
            step_start=0
            # begin with all image token ids masked
            mask_token_id = self.config.mask_token_id   # 58497
            num_vq_tokens = config.model.showo.num_vq_tokens    # 256
            num_new_special_tokens = config.model.showo.num_new_special_tokens  # 10

            input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
            input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id,
                                                        mask_token_id,
                                                        input_ids_minus_lm_vocab_size - config.model.showo.llm_vocab_size - num_new_special_tokens)

            # for classifier-free guidance
            if uncond_input_ids is not None:
                uncond_prefix = uncond_input_ids[:, :config.dataset.preprocessing.max_seq_length + 1]

        for step in range(step_start,timesteps):
            if uncond_prefix is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, config.dataset.preprocessing.max_seq_length + 1:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                with torch.enable_grad():
                    cond_logits, uncond_logits = self(model_input, attention_mask=attention_mask).chunk(2)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[:, -(num_vq_tokens + 1):-1, config.model.showo.llm_vocab_size + num_new_special_tokens:-1]
            else:
                with torch.enable_grad():
                    logits = self(input_ids, attention_mask=attention_mask)
                logits = logits[:, -(num_vq_tokens + 1):-1, config.model.showo.llm_vocab_size + num_new_special_tokens:-1]
            # print(cond_logits,uncond_logits,logits)
            # logits = torch.nan_to_num(logits, nan=-1e9, posinf=-1e9, neginf=-1e9)
            probs = logits.softmax(dim=-1)
            sampled = probs.reshape(-1, logits.size(-1))
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])
            # sampled_ids = torch.argmax(sampled, dim=-1).view(*logits.shape[:-1])

            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
            # Defines the mask ratio for the next round. The number to mask out is
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))
            # Computes the probabilities of each selected tokens.
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)

            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            # Gets mask lens for each sample in the batch according to the mask ratio.
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = torch.max(
                torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )

            if save_checkpoint and step == checkpoint_step:  # 关键：第9步触发，一次性填充所有掩码
                mask_len = torch.tensor([0], device=logits.device).unsqueeze(0)  # 掩码长度设为0，所有token均保留
                # 用当前step的预测结果覆盖所有剩余掩码（确保无遗漏）
                sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
                checkpoint = {
                        "step": step + 1, 
                        "mask_token_id": mask_token_id,
                        "num_vq_tokens": num_vq_tokens,
                        "num_new_special_tokens": num_new_special_tokens,
                        "input_ids": input_ids, 
                        "input_ids_minus_lm_vocab_size": input_ids_minus_lm_vocab_size,
                        "uncond_prefix": uncond_prefix,
                        "temperature": temperature * (1.0 - ratio), 
                        "model_parameters": list(self.parameters()),
                    }
                return sampled_ids,checkpoint
            
            # Adds noise for randomness
            temperature = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
            # Masks tokens with lower confidence.
            input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id,
                                                          sampled_ids + config.model.showo.llm_vocab_size
                                                          + num_new_special_tokens)
            input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)
        if return_logits:
            return sampled_ids,probs
        return sampled_ids
    
    
    def t2i_generate_return_multi_steps_result(
            self,
            steps_to_return: list = [],
            input_ids: torch.LongTensor = None,
            uncond_input_ids: torch.LongTensor = None,
            attention_mask=None,
            temperature=1.0,
            timesteps=18,  # ideal number of steps is 18 in maskgit paper
            guidance_scale=0,
            noise_schedule=cosine_schedule,
            generator: torch.Generator = None,
            config=None,
            **kwargs,
    ):
        # begin with all image token ids masked
        mask_token_id = self.config.mask_token_id   # 58497
        num_vq_tokens = config.model.showo.num_vq_tokens    # 256
        num_new_special_tokens = config.model.showo.num_new_special_tokens  # 10

        input_ids_minus_lm_vocab_size = input_ids[:, -(num_vq_tokens + 1):-1].clone()
        input_ids_minus_lm_vocab_size = torch.where(input_ids_minus_lm_vocab_size == mask_token_id,
                                                    mask_token_id,
                                                    input_ids_minus_lm_vocab_size - config.model.showo.llm_vocab_size - num_new_special_tokens)

        # for classifier-free guidance
        if uncond_input_ids is not None:
            uncond_prefix = uncond_input_ids[:, :config.dataset.preprocessing.max_seq_length + 1]

        steps_to_return.append(timesteps - 1)
        results_dict = {}
        
        for step in range(timesteps):
            if uncond_input_ids is not None and guidance_scale > 0:
                uncond_input_ids = torch.cat(
                    [uncond_prefix, input_ids[:, config.dataset.preprocessing.max_seq_length + 1:]], dim=1)
                model_input = torch.cat([input_ids, uncond_input_ids])
                cond_logits, uncond_logits = self(model_input, attention_mask=attention_mask).chunk(2)
                # logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
                # it seems that muse has a different cfg setting
                logits = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits
                logits = logits[:, -(num_vq_tokens + 1):-1, config.model.showo.llm_vocab_size + num_new_special_tokens:-1]
            else:
                logits = self(input_ids, attention_mask=attention_mask)
                logits = logits[:, -(num_vq_tokens + 1):-1, config.model.showo.llm_vocab_size + num_new_special_tokens:-1]

            probs = logits.softmax(dim=-1)
            sampled = probs.reshape(-1, logits.size(-1))
            sampled_ids = torch.multinomial(sampled, 1, generator=generator)[:, 0].view(*logits.shape[:-1])

            unknown_map = input_ids_minus_lm_vocab_size == mask_token_id
            sampled_ids = torch.where(unknown_map, sampled_ids, input_ids_minus_lm_vocab_size)
            
            with torch.no_grad():
                if step in steps_to_return:
                    results_dict[step] = sampled_ids.clone()
            
            # Defines the mask ratio for the next round. The number to mask out is
            # determined by mask_ratio * unknown_number_in_the_beginning.
            ratio = 1.0 * (step + 1) / timesteps
            mask_ratio = noise_schedule(torch.tensor(ratio))
            # Computes the probabilities of each selected tokens.
            selected_probs = torch.gather(probs, -1, sampled_ids.long()[..., None])
            selected_probs = selected_probs.squeeze(-1)

            # Ignores the tokens given in the input by overwriting their confidence.
            selected_probs = torch.where(unknown_map, selected_probs, torch.finfo(selected_probs.dtype).max)
            # Gets mask lens for each sample in the batch according to the mask ratio.
            mask_len = (num_vq_tokens * mask_ratio).floor().unsqueeze(0).to(logits.device)
            # Keeps at least one of prediction in this round and also masks out at least
            # one and for the next iteration
            mask_len = torch.max(
                torch.tensor([1], device=logits.device), torch.min(unknown_map.sum(dim=-1, keepdim=True) - 1, mask_len)
            )
            # Adds noise for randomness
            temperature = temperature * (1.0 - ratio)
            masking = mask_by_random_topk(mask_len, selected_probs, temperature, generator=generator)
            # Masks tokens with lower confidence.
            input_ids[:, -(num_vq_tokens + 1):-1] = torch.where(masking, mask_token_id,
                                                          sampled_ids + config.model.showo.llm_vocab_size
                                                          + num_new_special_tokens)
            input_ids_minus_lm_vocab_size = torch.where(masking, mask_token_id, sampled_ids)

        return results_dict


    @torch.no_grad()
    def mmu_generate(self, idx=None, input_embeddings=None, attention_mask=None, max_new_tokens=100, temperature=1.0, top_k=None, eot_token=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        try:
            device = idx.device
        except:
            device = input_embeddings.device
        # 在获取logits前添加（函数开头附近）
        if input_embeddings is not None:
            if torch.isnan(input_embeddings).any():
                print("❌ 输入embedding含nan")
            if torch.isinf(input_embeddings).any():
                print("❌ 输入embedding含inf")

        if attention_mask is not None:
            if torch.isnan(attention_mask).any():
                print("❌ 注意力掩码含nan")
            if torch.isinf(attention_mask).any():
                print("❌ 注意力掩码含inf")
        result = []
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            # idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            # logits, _ = self(idx_cond)
            logits = self(idx, input_embeddings=input_embeddings, attention_mask=attention_mask)

            L = attention_mask.shape[-1]
            attention_mask = attention_mask.squeeze()
            attention_mask_a = torch.hstack(
                [
                    attention_mask,  # L, L
                    torch.zeros((L, 1)).to(device) + torch.finfo(logits.dtype).min,
                ]
            )
            attention_mask_b = torch.vstack(
                [
                    attention_mask_a,  # L, L+1
                    torch.hstack([attention_mask[-1, :], torch.tensor([0]).to(device)]).unsqueeze(0),
                ]
            )
            attention_mask = attention_mask_b

            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # 新增logits检测
            if torch.isnan(logits).any():
                print(f"❌ logits（温度缩放后）含nan，数量：{torch.isnan(logits).sum().item()}")
                print(f"logits范围：{logits.min().item()} ~ {logits.max().item()}")
                print(input_embeddings)
            if torch.isinf(logits).any():
                print(f"❌ logits（温度缩放后）含inf，数量：{torch.isinf(logits).sum().item()}")
                print(f"logits范围：{logits.min().item()} ~ {logits.max().item()}")
            if (logits < -1e18).any():  # 极端负值可能导致softmax后为0，但需警惕
                print(f"❌ logits（温度缩放后）含极端负值，数量：{(logits < -1e18).sum().item()}")
            # logits = torch.nan_to_num(logits, nan=-1e9, posinf=-1e9, neginf=-1e9)
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                # if torch.isinf(logits).any():
                #     print(f"❌ logits（top_k裁剪后）含inf，数量：{torch.isinf(logits).sum().item()}")
                #     print(f"top_k后logits范围：{logits.min().item()} ~ {logits.max().item()}")

            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # 新增probs全维度检测
            if torch.isnan(probs).any():
                print(f"❌ probs（softmax后）含nan，数量：{torch.isnan(probs).sum().item()}")
            if torch.isinf(probs).any():
                print(f"❌ probs（softmax后）含inf，数量：{torch.isinf(probs).sum().item()}")
            if (probs < 0).any():
                print(f"❌ probs（softmax后）含负值，数量：{(probs < 0).sum().item()}")
                print(f"probs负值范围：{probs[probs < 0].min().item()} ~ {probs[probs < 0].max().item()}")
            if (probs.sum(dim=-1) < 0.9).any():  # softmax后每行和应≈1，若远小于1可能有异常
                print(f"❌ probs行和异常（<0.9），数量：{(probs.sum(dim=-1) < 0.9).sum().item()}")
                print(f"probs行和范围：{probs.sum(dim=-1).min().item()} ~ {probs.sum(dim=-1).max().item()}")
            # probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
            # probs = torch.clamp(probs, min=0.0)
            # 修复2：重新归一化概率（确保每行和为1，满足multinomial要求）
            # probs_sum = probs.sum(dim=-1, keepdim=True)
            # zero_sum_mask = probs_sum < 1e-9
            # 处理概率和为0的极端情况（用均匀分布兜底）
            # if torch.any(zero_sum_mask).item():
                # probs[zero_sum_mask] = 1.0 / probs.size(-1)
            # else:
                # probs = probs / probs_sum
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            result.append(idx_next[0][0])
            # append sampled index to the running sequence and continue
            if self.config.w_clip_vit:
                idx_next_embeddings = self.showo.model.embed_tokens(idx_next)
                input_embeddings = torch.cat([input_embeddings, idx_next_embeddings], dim=1)
            else:
                idx = torch.cat((idx, idx_next), dim=1)

            if eot_token is not None:
                # 确保比较在相同设备上，并且取出标量值
                if idx_next.to(device) == torch.tensor(eot_token, device=device):
                    break

        return result


