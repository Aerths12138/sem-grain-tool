
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
    clipLimit=2.0,
    tileGridSize=(8,8)
)

enhanced = clahe.apply(denoise)

# ====================== TopHat增强（核心） ======================

print("执行TopHat亮环增强...")

# TopHat核
kernel_tophat = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (15,15)
)

# TopHat
tophat = cv2.morphologyEx(
    enhanced,
    cv2.MORPH_TOPHAT,
    kernel_tophat
)

# ====================== 二值化 ======================

print("二值化...")

_, binary = cv2.threshold(
    tophat,
    25,
    255,
    cv2.THRESH_BINARY
)

# ====================== 闭运算补全亮环 ======================

kernel_close = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5,5)
)

# ====================== 定向边缘桥接（核心优化） ======================

# 横向桥接核
kernel_h = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (7,1)
)

# 纵向桥接核
kernel_v = cv2.getStructuringElement(
    cv2.MORPH_RECT,
    (1,7)
)

# 45度方向桥接核
kernel_d1 = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1]
], dtype=np.uint8)

# 135度方向桥接核
kernel_d2 = np.array([
    [0,0,1],
    [0,1,0],
    [1,0,0]
], dtype=np.uint8)

# 分方向闭合
bridge_h = cv2.morphologyEx(
    binary,
    cv2.MORPH_CLOSE,
    kernel_h
)

bridge_v = cv2.morphologyEx(
    binary,
    cv2.MORPH_CLOSE,
    kernel_v
)

bridge_d1 = cv2.morphologyEx(
    binary,
    cv2.MORPH_CLOSE,
    kernel_d1
)

bridge_d2 = cv2.morphologyEx(
    binary,
    cv2.MORPH_CLOSE,
    kernel_d2
)

# 合并桥接结果
closed = cv2.bitwise_or(bridge_h, bridge_v)
closed = cv2.bitwise_or(closed, bridge_d1)
closed = cv2.bitwise_or(closed, bridge_d2)


# ====================== 填充闭合区域内部 ======================

contours, hierarchy = cv2.findContours(
    closed,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

filled = np.zeros_like(closed)

for cnt in contours:

    area = cv2.contourArea(cnt)

    # 过滤小噪声
    if area > 30:

        cv2.drawContours(
            filled,
            [cnt],
            -1,
            255,
            thickness=cv2.FILLED
        )



# ====================== 温和去噪 ======================

median = cv2.medianBlur(
    filled,
    3
)

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    median,
    connectivity=8
)

result = np.zeros_like(median)

MIN_AREA = 20

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
plt.imshow(tophat, cmap='gray')
plt.title("TopHat增强")
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

