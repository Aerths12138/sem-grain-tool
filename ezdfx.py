import cv2
import numpy as np
import ezdxf
from pathlib import Path


# =========================
# 1. 路径设置
# =========================
input_image = r"voronoi_mixed_overlay.png"
output_dxf = r"voronoi_color_vectors.dxf"


# =========================
# 2. 读取图片
# =========================
img = cv2.imread(input_image)

if img is None:
    raise FileNotFoundError(f"无法读取图片：{input_image}")

h, w = img.shape[:2]

# BGR 转 HSV，便于按颜色分割
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


# =========================
# 3. 按颜色提取区域
# =========================
# 绿色线条范围
green_lower = np.array([35, 60, 60])
green_upper = np.array([90, 255, 255])
green_mask = cv2.inRange(hsv, green_lower, green_upper)

# 粉红/红色线条范围
# 红色在 HSV 中可能跨 0 度，因此分两段
red_lower1 = np.array([0, 40, 80])
red_upper1 = np.array([15, 255, 255])

red_lower2 = np.array([160, 40, 80])
red_upper2 = np.array([180, 255, 255])

red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
red_mask = cv2.bitwise_or(red_mask1, red_mask2)


# =========================
# 4. 形态学处理：去噪、连通线条
# =========================
kernel = np.ones((3, 3), np.uint8)

green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

green_mask = cv2.medianBlur(green_mask, 3)
red_mask = cv2.medianBlur(red_mask, 3)


# =========================
# 5. 查找轮廓
# =========================
def find_contours(mask):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )
    return contours


green_contours = find_contours(green_mask)
red_contours = find_contours(red_mask)


# =========================
# 6. 创建 DXF 文件
# =========================
doc = ezdxf.new("R2010")
msp = doc.modelspace()

# 创建图层
doc.layers.new(name="GREEN_LINES", dxfattribs={"color": 3})  # 绿色
doc.layers.new(name="RED_LINES", dxfattribs={"color": 1})    # 红色


# =========================
# 7. 轮廓写入 DXF
# =========================
def add_contours_to_dxf(contours, layer_name, simplify_epsilon=1.2, min_area=5):
    """
    contours: OpenCV 轮廓
    layer_name: DXF 图层名
    simplify_epsilon: 越大线条越简化，越小越贴近原图
    min_area: 过滤小噪点
    """

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < min_area:
            continue

        # 多边形简化，避免 DXF 点太多
        approx = cv2.approxPolyDP(cnt, simplify_epsilon, closed=True)

        points = []

        for p in approx:
            x, y = p[0]

            # DXF 坐标系 Y 轴向上，图片坐标系 Y 轴向下，所以这里翻转 y
            points.append((float(x), float(h - y)))

        if len(points) >= 2:
            # 闭合轮廓
            msp.add_lwpolyline(
                points,
                close=True,
                dxfattribs={
                    "layer": layer_name
                }
            )


add_contours_to_dxf(green_contours, "GREEN_LINES", simplify_epsilon=1.0, min_area=5)
add_contours_to_dxf(red_contours, "RED_LINES", simplify_epsilon=1.0, min_area=5)


# =========================
# 8. 保存 DXF
# =========================
doc.saveas(output_dxf)

print("转换完成：", output_dxf)
print("绿色轮廓数量：", len(green_contours))
print("红色轮廓数量：", len(red_contours))