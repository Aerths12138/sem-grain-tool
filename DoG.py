
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff

# ====================== 路径 ======================

IMG_PATH = r"1_i313.tif"

SAVE_PATH = r"E:\deskup\工作\sem识图\result.tif"

# =================================================

print("读取SEM图...")

img = tiff.imread(IMG_PATH)

# ---------- 16位转8位 ----------

if img.dtype == np.uint16:

    img = cv2.normalize(
        img,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
        dtype=cv2.CV_8U
    )

# ---------- 转灰度 ----------

if len(img.shape) == 3:

    img = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

# ====================== SEM增强 ======================

print("进行SEM增强...")

# 非局部去噪
denoise = cv2.fastNlMeansDenoising(
    img,
    h=5
)

# CLAHE增强
clahe = cv2.createCLAHE(
    clipLimit=2.5,
    tileGridSize=(8,8)
)

enhanced = clahe.apply(denoise)

# ====================== DoG增强（核心） ======================

print("执行DoG边缘增强...")

# 小尺度高斯模糊
blur_small = cv2.GaussianBlur(
    enhanced,
    (0,0),
    sigmaX=1
)

# 大尺度高斯模糊
blur_large = cv2.GaussianBlur(
    enhanced,
    (0,0),
    sigmaX=5
)

# DoG
dog = cv2.subtract(
    blur_small,
    blur_large
)

# 归一化
dog = cv2.normalize(
    dog,
    None,
    0,
    255,
    cv2.NORM_MINMAX
).astype(np.uint8)

# ====================== 二值化 ======================

print("二值化...")

_, binary = cv2.threshold(
    dog,
    40,
    255,
    cv2.THRESH_BINARY
)

# ====================== 轻度闭运算 ======================

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (3,3)
)

closed = cv2.morphologyEx(
    binary,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=1
)

# ====================== 温和去噪 ======================

# 中值滤波
denoise2 = cv2.medianBlur(
    closed,
    3
)

# 轻度面积过滤
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    denoise2,
    connectivity=8
)

result = np.zeros_like(denoise2)

MIN_AREA = 8

for i in range(1, num_labels):

    area = stats[i, cv2.CC_STAT_AREA]

    if area >= MIN_AREA:

        result[labels == i] = 255

# ====================== 保存 ======================

tiff.imwrite(
    SAVE_PATH,
    result
)

print("完成！")

# ====================== 显示 ======================

plt.figure(figsize=(20,5))

plt.subplot(151)
plt.imshow(img, cmap='gray')
plt.title("原始SEM")
plt.axis("off")

plt.subplot(152)
plt.imshow(enhanced, cmap='gray')
plt.title("CLAHE增强")
plt.axis("off")

plt.subplot(153)
plt.imshow(dog, cmap='gray')
plt.title("DoG结果")
plt.axis("off")

plt.subplot(154)
plt.imshow(binary, cmap='gray')
plt.title("二值化")
plt.axis("off")

plt.subplot(155)
plt.imshow(result, cmap='gray')
plt.title("最终结果")
plt.axis("off")

plt.tight_layout()
plt.show()
