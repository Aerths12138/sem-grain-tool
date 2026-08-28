import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from PIL import Image
from transformers import SamModel, SamProcessor
import torch
import warnings
warnings.filterwarnings('ignore')

# ====================== 你只需要改这里 ======================
TIF_FILE_PATH = r"1_i311.tif"
SAVE_RESULT_PATH = r"E:\deskup\工作\sem识图\结果.tif"
# ==========================================================

# 读取图片
img = tiff.imread(TIF_FILE_PATH)

if img.dtype == np.uint16:
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
if len(img.shape) == 3:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# 预处理
denoised = cv2.fastNlMeansDenoising(img, h=3)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
enhanced = clahe.apply(denoised)
input_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

# 模型
device = "cpu"
model = SamModel.from_pretrained("dhkim2810/MobileSAM").to(device)
processor = SamProcessor.from_pretrained("dhkim2810/MobileSAM", use_fast=False)

# 分割
inputs = processor(Image.fromarray(input_rgb), points_per_batch=32, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

# 掩码
masks = processor.post_process_masks(
    outputs.pred_masks.cpu(),
    inputs["original_sizes"].cpu(),
    inputs["reshaped_input_sizes"].cpu()
)[0]

result = np.zeros_like(img)
for mask in masks:
    mask = mask.squeeze()
    if mask.sum() > 50:
        result[mask] = 255

# 填充孔洞
contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in contours:
    cv2.drawContours(result, [cnt], -1, 255, thickness=cv2.FILLED)

# 保存
tiff.imwrite(SAVE_RESULT_PATH, result)
print("✅ 完成！结果已保存到:", SAVE_RESULT_PATH)

# 显示
plt.figure(figsize=(12,5))
plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('原图')
plt.subplot(132), plt.imshow(enhanced, cmap='gray'), plt.title('增强')
plt.subplot(133), plt.imshow(result, cmap='gray'), plt.title('结果')
plt.show()