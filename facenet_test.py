import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from feat.identity_detectors.facenet.facenet_model import InceptionResnetV1

# from utils import *

# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
from facenet_pytorch import MTCNN, InceptionResnetV1
import matplotlib.pyplot as plt
# 初始化模型
identity_detector = InceptionResnetV1(
    pretrained=None,
    classify=False,
    num_classes=None,
    dropout_prob=0.6,
    device=device,
)
identity_detector.logits = nn.Linear(512, 8631)

# 加载模型权重
identity_model_file='/root/.cache/huggingface/hub/models--py-feat--facenet/snapshots/c42f4c2771b043bb5c461093664d016ff4298224/facenet_20180402_114759_vggface2.pth'
identity_detector.load_state_dict(torch.load(identity_model_file, map_location=device))
identity_detector.eval()
identity_detector.to(device)
# 初始化 MTCNN 模型（用于检测和对齐）
# 注意：keep_all=True 表示保留所有检测到的人脸，default_landmarks=True 输出关键点
mtcnn = MTCNN(
    image_size=160,  # 对齐后的人脸尺寸（与 Facenet 输入匹配）
    margin=32,       # 裁剪时的边缘余量
    min_face_size=20,  # 最小可检测人脸尺寸
    thresholds=[0.6, 0.7, 0.7],  # 三级网络的置信度阈值
    factor=0.709,    # 图像金字塔缩放因子
    post_process=True,  # 对齐后是否标准化
    device='cuda' if torch.cuda.is_available() else 'cpu'  # 设备
)
image_path1 = "/share/project/emllm_mnt.1d/mnt/hpfs/baaiei/daigaole/code/UnicR1/dataset/unictokens_data/concept/test/adrien_brody/2.png"
image_path2 = "/share/project/emllm_mnt.1d/mnt/hpfs/baaiei/daigaole/code/UnicR1/showo/t2i_saved/adrien_brody/15/epoch=2/A photo of <adrien_brody> running./18.png"
# 加载示例图像
# image_path = "/share/project/emllm_mnt.1d/mnt/hpfs/baaiei/daigaole/code/UnicR1/showo/tmp_result/best_image/adrien_brody/image_000005.png"  # 替换为你的图像路径
img1 = Image.open(image_path1).convert('RGB')
img2 = Image.open(image_path2).convert('RGB')

# 1. 检测人脸并获取对齐后的图像
# 返回：对齐后的人脸图像（若检测到多人，返回列表）
face_tensor1 = mtcnn(img1)
face_tensor2 = mtcnn(img2)
with torch.no_grad():  # 关闭梯度计算，节省内存并加速
    identity_embeddings1 = identity_detector(face_tensor1.unsqueeze(0).to(device))
with torch.no_grad():  # 关闭梯度计算，节省内存并加速
    identity_embeddings2 = identity_detector(face_tensor2.unsqueeze(0).to(device))
print(f"distance: {torch.norm(identity_embeddings1 - identity_embeddings2)}")
cos_sim = torch.nn.functional.cosine_similarity(identity_embeddings1, identity_embeddings2)  # 相似度
cos_dist = 1 - cos_sim
print(f"cosine distance",cos_dist.item())
# 2. 若需要查看关键点（可选）
# 重新运行检测，获取边界框和关键点
# boxes1, probs1, landmarks1 = mtcnn.detect(img1, landmarks=True)
# boxes2, probs2, landmarks2 = mtcnn.detect(img2, landmarks=True)

# 3. 可视化结果
# plt.figure(figsize=(12, 6))

# # 原图 + 关键点标记
# plt.subplot(121)
# plt.imshow(img)
# if landmarks is not None:
#     # 绘制关键点（眼睛、鼻子、嘴巴等）
#     for landmark in landmarks:
#         plt.scatter(landmark[:, 0], landmark[:, 1], s=50, c='red', marker='o')
# plt.title("Original Image with Landmarks")

# # 对齐后的人脸
# plt.subplot(122)
# if aligned_face is not None:
#     # 关键步骤1：将 [-1, 1] 映射到 [0, 255]
#     # 公式：(x + 1) / 2 * 255 → 把 [-1,1] 先转为 [0,1]，再放大到 [0,255]
#     aligned_img_norm = (aligned_face + 1) / 2 * 255  
#     # 关键步骤2：转为 uint8 类型（图像显示必须），并调整维度（C,H,W → H,W,C）
#     aligned_img = aligned_img_norm.permute(1, 2, 0).cpu().numpy().astype(np.uint8)  
#     plt.imshow(aligned_img)
#     plt.title("Aligned Face (160x160)")
# else:
#     plt.title("No face detected")

# plt.savefig("./")


# 定义图像预处理转换 - 符合Facenet模型的输入要求
# preprocess = transforms.Compose([
#     transforms.Resize((160, 160)),  # Facenet通常需要160x160的输入
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化
# ])
# "/share/project/emllm_mnt.1d/mnt/hpfs/baaiei/daigaole/code/UnicR1/showo/tmp_result/best_image/adrien_brody/image_000005.png"
# 测试模型

# 加载并预处理图像
# try:
#     # 打开图像并转换为RGB格式
#     image1 = Image.open(image_path1).convert('RGB')
#     # 应用预处理
#     face_tensor1 = preprocess(image1)
#     # 添加批次维度 (模型通常期望批次输入)
#     face_tensor1 = face_tensor1.unsqueeze(0).to(device)
    
#     # 提取特征
#     with torch.no_grad():  # 关闭梯度计算，节省内存并加速
#         identity_embeddings1 = identity_detector(face_tensor1)

#     image2 = Image.open(image_path2).convert('RGB')
#     # 应用预处理
#     face_tensor2 = preprocess(image2)
#     # 添加批次维度 (模型通常期望批次输入)
#     face_tensor2 = face_tensor2.unsqueeze(0).to(device)
    
#     # 提取特征
#     with torch.no_grad():  # 关闭梯度计算，节省内存并加速
#         identity_embeddings2 = identity_detector(face_tensor2)
#     print(f"distance: {torch.norm(identity_embeddings1 - identity_embeddings2)}")
#     cos_sim = torch.nn.functional.cosine_similarity(identity_embeddings1, identity_embeddings2)  # 相似度
#     cos_dist = 1 - cos_sim
#     print(f"cosine distance",cos_dist)
# except Exception as e:
#     print(f"处理过程出错: {str(e)}")
