from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =====================================
# 读取二值图
# =====================================
img = cv2.imread(r"output_result.png", cv2.IMREAD_GRAYSCALE)
binary = img > 127

# =====================================
# 距离变换
# =====================================
distance = ndimage.distance_transform_edt(binary)

# =====================================
# 自动寻找种子点
# =====================================
coords = peak_local_max(distance, min_distance=35, labels=binary)
markers = np.zeros_like(distance, dtype=np.int32)
for i, (y, x) in enumerate(coords):
    markers[y, x] = i + 1

# =====================================
# Watershed 分割
# =====================================
labels = watershed(-distance, markers, mask=binary)
print("颗粒数量:", labels.max())

# 显示 Watershed 分割结果
plt.figure(figsize=(10, 10))
plt.imshow(labels, cmap="nipy_spectral")
plt.title(f"Watershed Particles: {labels.max()}")
plt.axis("off")
plt.show()

# =====================================
# 构建封闭曲线图
# =====================================
canvas = np.zeros(img.shape, dtype=np.uint8)

for lbl in range(1, labels.max() + 1):
    mask = np.uint8(labels == lbl) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10:
            continue

        # 平滑轮廓
        epsilon = 0.003 * cv2.arcLength(cnt, True)
        smooth_cnt = cv2.approxPolyDP(cnt, epsilon, True)

        # 自动线宽（根据颗粒面积）
        thickness = max(2, min(5, int(np.sqrt(area) / 20)))

        cv2.drawContours(canvas, [smooth_cnt], -1, 255, thickness)

# =====================================
# 额外圆润处理
# =====================================
canvas = cv2.GaussianBlur(canvas, (5, 5), 0)
_, canvas = cv2.threshold(canvas, 100, 255, cv2.THRESH_BINARY)

# =====================================
# 显示最终结果
# =====================================
plt.figure(figsize=(10, 10))
plt.imshow(canvas, cmap="gray")
plt.title("Closed Curves Structure")
plt.axis("off")
plt.show()

# =====================================
# 保存结果
# =====================================
cv2.imwrite("watershed_closed_curves.png", canvas)
print("结果已保存：watershed_closed_curves.png")