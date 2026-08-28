from scipy import ndimage
from scipy.spatial import Voronoi
import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

# =====================================
# 核心最小值限制（调这几个参数就够了）
# =====================================
# 【最关键】种子点之间的最小像素距离，值越大种子越少、颗粒越大
# 强制最小值：低于30会自动修正，防止点过多
MIN_SEED_DISTANCE = 50
MIN_PARTICLE_AREA = 250
MIN_DIST_TO_EDGE = 12
MAX_SEED_COUNT = 200

# =====================================
# 全局配置
# =====================================
plt.rcParams["font.sans-serif"] = ["SimHei", "PingFang SC", "WenQuanYi Micro Hei"]
plt.rcParams["axes.unicode_minus"] = False

# 强制最小值校验：防止手动设太小导致点爆炸
MIN_SEED_DISTANCE = max(MIN_SEED_DISTANCE, 30)

# =====================================
# 1. 读取图像 + 预处理
# =====================================
img = cv2.imread(r"1_i311png.png", cv2.IMREAD_GRAYSCALE)
h, w = img.shape
binary = img > 127  # 白色颗粒=前景True，黑色背景=背景False

# =====================================
# 2. 带多层限制的种子点生成（核心修复）
# =====================================
# 距离变换：计算每个像素到最近背景的距离
distance = ndimage.distance_transform_edt(binary)

# 第一步：网格采样，基础保证种子点间距不小于最小值
seeds = []
step = MIN_SEED_DISTANCE
for y in range(step // 2, h, step):
    for x in range(step // 2, w, step):
        # 限制1：只在颗粒前景区域生成种子
        # 限制2：种子点离边缘至少MIN_DIST_TO_EDGE像素
        if binary[y, x] and distance[y, x] >= MIN_DIST_TO_EDGE:
            seeds.append([x, y])

seeds = np.array(seeds)

# 第二步：二次距离过滤，强制保证所有种子点间距≥MIN_SEED_DISTANCE
filtered_seeds = []
for x, y in seeds:
    # 检查和已保留种子的距离
    too_close = False
    for fx, fy in filtered_seeds:
        if np.hypot(x - fx, y - fy) < MIN_SEED_DISTANCE:
            too_close = True
            break
    if not too_close:
        filtered_seeds.append([x, y])

# 第三步：数量上限限制，超过最大数量就自动增大间距重采样
if len(filtered_seeds) > MAX_SEED_COUNT:
    print(f"种子点数量{len(filtered_seeds)}超过上限{MAX_SEED_COUNT}，自动调整最小间距")
    MIN_SEED_DISTANCE = int(MIN_SEED_DISTANCE * 1.3)
    # 重新采样（递归简化处理）
    filtered_seeds = []
    step = MIN_SEED_DISTANCE
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            if binary[y, x] and distance[y, x] >= MIN_DIST_TO_EDGE:
                filtered_seeds.append([x, y])

seeds = np.array(filtered_seeds)
print(f"最终有效种子点数量：{len(seeds)}，最小间距：{MIN_SEED_DISTANCE}像素")

# =====================================
# 3. 生成标准泰森多边形
# =====================================
# 边界虚拟种子，避免边缘多边形无限延伸
border_seeds = np.array([
    [-1000, -1000], [w + 1000, -1000], [-1000, h + 1000], [w + 1000, h + 1000],
    [-1000, h // 2], [w + 1000, h // 2], [w // 2, -1000], [w // 2, h + 1000]
])
all_seeds = np.vstack([seeds, border_seeds])
vor = Voronoi(all_seeds)

# =====================================
# 4. 带最小面积过滤的颗粒提取
# =====================================
fig, ax = plt.subplots(figsize=(12, 10), dpi=100)

# 底层：原始二值图
ax.imshow(img, cmap="gray")

valid_particles = []
# 遍历每个种子点对应的泰森多边形
for i, region_idx in enumerate(vor.point_region[:len(seeds)]):
    region = vor.regions[region_idx]
    # 跳过无效区域
    if -1 in region or len(region) < 3:
        continue

    # 提取多边形顶点，裁剪到图像范围内
    poly_points = vor.vertices[region]
    poly_points[:, 0] = np.clip(poly_points[:, 0], 0, w)
    poly_points[:, 1] = np.clip(poly_points[:, 1], 0, h)

    poly = Polygon(poly_points)

    # 【最小值限制】过滤面积过小的颗粒
    if poly.area < MIN_PARTICLE_AREA:
        continue

    valid_particles.append(poly)

    # 绘制：红色=泰森分割边界，绿色=颗粒轮廓
    x, y = poly.exterior.xy
    ax.plot(x, y, c="red", linewidth=0.8)
    # 向内收缩生成颗粒轮廓（可选）
    shrunk = poly.buffer(-2, join_style=1)
    if not shrunk.is_empty:
        sx, sy = shrunk.exterior.xy
        ax.plot(sx, sy, c="lime", linewidth=1)

# 绘制蓝色种子点
ax.scatter(seeds[:, 0], seeds[:, 1], c="blue", s=15, label="种子点")

# =====================================
# 5. 输出结果
# =====================================
ax.set_title(f"SEM标准泰森多边形分割\n颗粒总数：{len(valid_particles)}", fontsize=14)
ax.legend(loc="upper right")
ax.set_xlim(0, w)
ax.set_ylim(h, 0)
ax.axis("off")
plt.tight_layout()

plt.savefig("优化后泰森分割结果.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"最终有效颗粒总数：{len(valid_particles)}")
print(f"已强制限制：最小种子间距{MIN_SEED_DISTANCE}px，最小颗粒面积{MIN_PARTICLE_AREA}px²，最大种子数{MAX_SEED_COUNT}")