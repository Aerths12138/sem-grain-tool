
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff

# ====================== 路径 ======================

IMG_PATH = r"1_i311.tif"

SAVE_PATH = r"E:\deskup\工作\sem识图\result.tif"

# =================================================

print("读取SEM图...")

img = tiff.imread(IMG_PATH)

# 16位转8位
if img.dtype == np.uint16:
    img = cv2.normalize(
        img,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
        dtype=cv2.CV_8U
    )

# 转灰度
if len(img.shape) == 3:
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# ====================== 去噪 ======================

denoise = cv2.fastNlMeansDenoising(
    img,
    h=5
)

# ====================== SEM增强 ======================

clahe = cv2.createCLAHE(
    clipLimit=3.0,
    tileGridSize=(8,8)
)

enhanced = clahe.apply(denoise)

# ====================== 自适应阈值 ======================

binary = cv2.adaptiveThreshold(
    enhanced,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    31,
    -5
)

# ====================== 闭运算（关键） ======================

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5,5)
)

closed = cv2.morphologyEx(
    binary,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=2
)


# ====================== 使用轮廓填充内部区域 ======================

# 查找闭合轮廓
contours, hierarchy = cv2.findContours(
    closed,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# 创建结果图
filled = np.zeros_like(closed)

# 填充闭合区域
for cnt in contours:

    area = cv2.contourArea(cnt)

    # 过滤极小区域
    if area > 20:

        cv2.drawContours(
            filled,
            [cnt],
            -1,
            255,
            thickness=cv2.FILLED
        )




# ====================== 去小噪声（增强版） ======================

# ---------- 第一步：连通域面积过滤 ----------

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    filled,
    connectivity=8
)

result = np.zeros_like(filled)

# 最小保留面积（非常关键）
# 建议：
# 50 = 保守
# 100 = 推荐
# 200 = 强力去噪
MIN_AREA = 1

for i in range(1, num_labels):

    area = stats[i, cv2.CC_STAT_AREA]

    if area > MIN_AREA:

        result[labels == i] = 255



# ====================== 温和SEM去噪 ======================

# ---------- 方法1：中值滤波（推荐） ----------

# 专门去孤立白点
# 不会像开运算一样破坏边缘

denoise = cv2.medianBlur(closed, 3)

# ---------- 方法2：极轻微面积过滤 ----------

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    denoise,
    connectivity=8
)

result = np.zeros_like(denoise)

# 非常小即可
MIN_AREA = 200

for i in range(1, num_labels):

    area = stats[i, cv2.CC_STAT_AREA]

    if area >= MIN_AREA:

        result[labels == i] = 255



# ====================== 保存 ======================

tiff.imwrite(SAVE_PATH, result)

print("完成！")

# ====================== 显示 ======================

plt.figure(figsize=(18,6))

plt.subplot(141)
plt.imshow(img, cmap='gray')
plt.title("原始SEM")

plt.subplot(142)
plt.imshow(enhanced, cmap='gray')
plt.title("增强后")

plt.subplot(143)
plt.imshow(closed, cmap='gray')
plt.title("闭运算后")

plt.subplot(144)
plt.imshow(result, cmap='gray')
plt.title("最终分割")

for i in range(1,5):
    plt.subplot(1,4,i)
    plt.axis("off")

plt.tight_layout()
plt.show()
