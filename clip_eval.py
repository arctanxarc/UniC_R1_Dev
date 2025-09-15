import clip
import torch
import os
import numpy as np
import json
from PIL import Image
from torchvision import transforms
# from utils import check_dtype

class CLIPEvaluator(object):
    def __init__(self, device, clip_model='ViT-B/32',pretrained=None) -> None:
        self.device = device
        os.environ['CLIP_CACHE_DIR'] = os.path.dirname(pretrained) if pretrained else ''
        self.model, clip_preprocess = clip.load(pretrained, device=self.device)
        model_weights=pretrained
        # if os.path.exists(model_weights):
            # state_dict = torch.load(model_weights,map_location=device, weights_only=False)
            # print(state_dict.keys())
            # model = clip.build_model(clip.load_state_dict(torch.load(pretrained,map_location=device, weights_only=False))).to(device)
            # preprocess = clip._transform(model.input_resolution)
            # self.model=model
            # self.model = torch.jit.load(pretrained, map_location=device).eval()
        # 获取原始CLIP预处理
            # _, clip_preprocess = clip.load(clip_model, device=device, jit=False)
            
            # self.model.load_state_dict(state_dict)
        self.clip_preprocess = clip_preprocess
        
        self.preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (generator output) to [0, 1].
                                              clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
                                              clip_preprocess.transforms[4:])                                       # + skip convert PIL to tensor

    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)

    @torch.no_grad()
    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.preprocess(images).to(self.device)
        return self.model.encode_image(images)

    def get_text_features(self, text: str, norm: bool = True) -> torch.Tensor:

        tokens = clip.tokenize(text).to(self.device)

        text_features = self.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)
        
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    def img_to_img_similarity(self, src_images, generated_images):
        src_img_features = self.get_image_features(src_images)
        gen_img_features = self.get_image_features(generated_images)

        return (src_img_features @ gen_img_features.T).mean()

    def txt_to_img_similarity(self, text, generated_images):
        text_features    = self.get_text_features(text)
        gen_img_features = self.get_image_features(generated_images)

        return (text_features @ gen_img_features.T).mean()
    
    def evaluate(self, src_images, pred_images, target_text):
        sim_samples_to_img  = self.img_to_img_similarity(src_images, pred_images)
        sim_samples_to_text = self.txt_to_img_similarity(target_text, pred_images)

        return sim_samples_to_img, sim_samples_to_text
    

class SHOWO_P_CLIPEvaluator(CLIPEvaluator):
    def __init__(self, 
                 device, 
                 clip_model='/home/daigaole/code/ex/ViT-B-32.pt',
                 data_root='/home/daigaole/code/ex/dataset/unictokens_data/',
                 work_dir='/home/daigaole/code/ex/showo_feat',
                 save_dir='0.png',
                 dtype=None,
                 resized_size=512) -> None:
        # os.environ['OPEN_CLIP_NO_NETWORK'] = '1'
        super().__init__(device,pretrained=clip_model)
        self.data_root = data_root
        self.gen_saved_dir = os.path.join(work_dir, save_dir)
        self.results = {}
        self.resized_size = resized_size
        self.save_dir=save_dir
        # self.model=check_dtype(self.model,dtype)
        
    def image_transform(self, image):
        image = transforms.Resize(self.resized_size, interpolation=transforms.InterpolationMode.BICUBIC)(image)
        image = transforms.CenterCrop(size=(self.resized_size, self.resized_size))(image)
        image = transforms.ToTensor()(image)
        return image  
    
    def evaluate_concept_ref2ref(self, concept):
        
        ref_images_dir = os.path.join(self.data_root, "concept/train", concept)
        
        train_images_paths_file = os.path.join(self.data_root, "concept/train/train_images.json")
        with open(train_images_paths_file, 'r') as f:
            train_images_paths = json.load(f)
        train_images_paths_for_concept = train_images_paths[concept]
        src_images_path = [os.path.join(ref_images_dir, img_path) for img_path in train_images_paths_for_concept]
        src_images = [Image.open(img_path).convert("RGB") for img_path in src_images_path]
        # convert Image to tensor
        src_images_tensor = [self.image_transform(img) for img in src_images]
        # Resize images to the same size
        
        src_images_tensor = torch.stack(src_images_tensor).to(self.device)  # (N, 3, H, W)
        
        sims = []
        for src_img_tensor in src_images_tensor:
            img_tensor = src_img_tensor.unsqueeze(0)
            # Evaluate similarity
            similarity = self.img_to_img_similarity(src_images_tensor, img_tensor).item()
            sims.append(similarity)
        avg_similarity = sum(sims) / len(sims)
        return avg_similarity
    
    def evaluate_concept(self, concept, ckpt_name, epoch2load):
        
        self.results[concept] = {}
        
        ref_images_dir = os.path.join(self.data_root, "concept/train", concept)
        
        train_images_paths_file = os.path.join(self.data_root, "concept/train/train_images.json")
        with open(train_images_paths_file, 'r') as f:
            train_images_paths = json.load(f)
        train_images_paths_for_concept = train_images_paths[concept]
        src_images_path = [os.path.join(ref_images_dir, img_path) for img_path in train_images_paths_for_concept]
        src_images = [Image.open(img_path).convert("RGB") for img_path in src_images_path]
        # convert Image to tensor
        src_images_tensor = [self.image_transform(img) for img in src_images]
        src_images_tensor = torch.stack(src_images_tensor).to(self.device)  # (N, 3, H, W)
        
        # concept_info_file = os.path.join(self.data_root, "concept/train", concept, "info.json")
        # with open(concept_info_file, 'r') as f:
        #     concept_info = json.load(f)
        # concept_class = concept_info['class']
        
        # gened_prompts_dir = os.path.join(self.gen_saved_dir, concept, ckpt_name, f"{epoch2load}")
        # prompts_to_eval = [dir_name for dir_name in os.listdir(gened_prompts_dir) if os.path.isdir(os.path.join(gened_prompts_dir, dir_name))]
        # print(f"Prompts to eval: {prompts_to_eval}")
        
        # for prompt in prompts_to_eval:
        #     self.results[concept][prompt] = {}
        #     for img in os.listdir(os.path.join(gened_prompts_dir, prompt)):
        #         if not img.endswith('.png'):
        #             continue
        #         img_path = os.path.join(gened_prompts_dir, prompt, img)
        #         generated_image = Image.open(img_path).convert("RGB")
        #         generated_image_tensor = self.image_transform(generated_image).unsqueeze(0).to(self.device)
        #         # Evaluate similarity
        #         sim_samples_to_img, sim_samples_to_text = self.evaluate(src_images_tensor, generated_image_tensor, prompt.replace('<sks>', f'{concept_class}'))
        #         self.results[concept][prompt][img] = {
        #             'sim_samples_to_img': sim_samples_to_img.item(),
        #             'sim_samples_to_text': sim_samples_to_text.item()
        #         }
        img_path=self.save_dir
        generated_image = Image.open(img_path).convert("RGB")
        generated_image_tensor = self.image_transform(generated_image).unsqueeze(0).to(self.device)
        sim_samples_to_img, _ = self.evaluate(src_images_tensor, generated_image_tensor, '')
        return sim_samples_to_img

    def save_results(self, save_path):
        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"Results saved to {save_path}")
        
    def print_avg_results(self):
        # print global average results of all concepts
        img2img = []
        img2text = []

        for concept, prompt_item in self.results.items():
            if concept not in concepts:
                continue
            for prompt, item in prompt_item.items():
                for img, scores in item.items():
                    img2img.append(scores["sim_samples_to_img"])
                    img2text.append(scores["sim_samples_to_text"])

        img2img_avg = sum(img2img) / len(img2img)
        img2text_avg = sum(img2text) / len(img2text)
        print(f"img2img_avg: {img2img_avg}")
        print(f"img2text_avg: {img2text_avg}")
        

if __name__ == '__main__':
    work_dir = "/home/hpyky/Show-o"
    
    model = SHOWO_P_CLIPEvaluator("cuda:7", work_dir=work_dir, save_dir="gen_saved")
    ckpt_name = "showo_prompt"
    epoch2load = 0
    concept_list_file = "/home/hpyky/show_data/concepts_list.json"
    with open(concept_list_file, 'r') as f:
        concepts = json.load(f)
    concepts = ["adrien_brody", "bingbing", "bo", "coco", "dunpai", "gold_pineapple", "leonardo", "wangkai", "willinvietnam"]
    sims = []
    for concept in concepts:
        # sim = model.evaluate_concept_ref2ref(concept)
        # sims.append(sim)
        # print(f"Concept: {concept}, Similarity: {sim}")
        model.evaluate_concept(concept, ckpt_name, epoch2load)
    model.save_results(f"gen1_clip_eval_results_{ckpt_name}_{epoch2load}.json")
    model.print_avg_results()
    # avg_sim = sum(sims) / len(sims)
    # print(f"Average Similarity: {avg_sim}")
    
    
    
