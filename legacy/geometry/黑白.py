import cv2
import numpy as np
import ezdxf

# -------------------------- 可调整参数 --------------------------
# 输入图片路径
input_image_path = "voronoi_mixed_overlay.png"
# 输出DXF文件路径
output_dxf_path = "sem_grain.dxf"
# 饱和度阈值（和之前的图像处理逻辑一致）
saturation_threshold = 10
# 轮廓最小面积过滤（单位：像素，去掉太小的噪点）
min_contour_area = 20
# 轮廓简化精度（值越大轮廓越平滑，文件越小；0则保留原始像素轮廓）
contour_epsilon = 0.5
# 坐标缩放比例（1像素对应CAD中的单位长度，比如1像素=0.01mm，可根据标尺调整）
# 你的图标尺是50um对应图中像素长度，可自行校准
scale = 1
# ----------------------------------------------------------------

# ========== 第一步：和之前一致，生成黑白二值图 ==========
img = cv2.imread(input_image_path)
if img is None:
    raise FileNotFoundError(f"无法读取图片，请检查路径: {input_image_path}")
img_h, img_w = img.shape[:2]

# 转HSV提取饱和度，生成二值mask
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
s_channel = hsv_img[:, :, 1]
_, binary_mask = cv2.threshold(s_channel, saturation_threshold, 255, cv2.THRESH_BINARY)

# 可选：去噪点，让轮廓更干净
kernel = np.ones((2, 2), np.uint8)
binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

# ========== 第二步：提取白色区域的轮廓 ==========
# 只提取最外层轮廓，避免内部孔洞
contours, _ = cv2.findContours(
    binary_mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# 过滤小面积噪点 + 简化轮廓
valid_contours = []
for cnt in contours:
    # 过滤面积太小的噪点
    if cv2.contourArea(cnt) < min_contour_area:
        continue
    # 简化轮廓，减少顶点数量
    approx_cnt = cv2.approxPolyDP(cnt, epsilon=contour_epsilon, closed=True)
    valid_contours.append(approx_cnt)

print(f"共提取到 {len(valid_contours)} 个有效颗粒轮廓")

# ========== 第三步：生成DXF矢量文件 ==========
# 创建DXF文档（默认R2010版本，兼容所有CAD软件）
doc = ezdxf.new(dxfversion="R2010")
msp = doc.modelspace()

for cnt in valid_contours:
    # 转换轮廓坐标：图像原点是左上角，CAD原点是左下角，需要翻转Y轴
    points = []
    for point in cnt:
        x = point[0][0] * scale
        # Y轴翻转：图像高度 - 原始Y坐标
        y = (img_h - point[0][1]) * scale
        points.append((x, y))

    # 在DXF中添加闭合多段线（每个颗粒是一个独立闭合轮廓）
    msp.add_lwpolyline(points, close=True)

# 保存DXF文件
doc.saveas(output_dxf_path)
print(f"DXF文件已生成，保存路径: {output_dxf_path}")