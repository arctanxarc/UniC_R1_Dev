import face_recognition
def face_recognition_score(known_image_path, generated_image_path):
    # 加载图像并获取人脸编码
    known_image = face_recognition.load_image_file(known_image_path)
    generated_image = face_recognition.load_image_file(generated_image_path)
    
    # 提取人脸编码（假设每张图只有一个人脸）
    try:
        known_encoding = face_recognition.face_encodings(known_image)[0]
        generated_encoding = face_recognition.face_encodings(generated_image)[0]
    except IndexError:
        return "无法在图像中检测到人脸"
    
    # 计算人脸距离（越小越相似）
    distance = face_recognition.face_distance([known_encoding], generated_encoding)[0]
    
    # 将距离转换为0-100的分数（距离0→100分，距离0.6→0分）
    max_distance = 1  # 默认阈值
    if distance >= max_distance:
        return 0.0
    score = 100.0 * (1 - distance / max_distance)
    return round(score/100, 2)

img1='/share/project/emllm_mnt.1d/mnt/hpfs/baaiei/daigaole/code/UnicR1/dataset/unictokens_data/concept/test/bo/0.png'
img2='/share/project/emllm_mnt.1d/mnt/hpfs/baaiei/daigaole/code/UnicR1/dataset/unictokens_data/concept/test/bo/2.png'
print(face_recognition_score(img1,img2))