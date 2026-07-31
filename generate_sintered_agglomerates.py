import argparse
import csv
import math
from pathlib import Path

import cv2
import ezdxf
import numpy as np
from PIL import Image
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.ops import unary_union


ROOT = Path(__file__).resolve().parent


def save_png(path: Path, image: np.ndarray) -> None:
    """OpenCV 在中文路径下直接 imwrite 可能失败，这里用编码后写文件。"""
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Could not encode PNG: {path}")
    encoded.tofile(str(path))


def read_particles(path: Path, min_area: float) -> list[dict[str, float]]:
    """读取颗粒统计 CSV，只保留面积大于 min_area 的颗粒。"""
    particles: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            area = float(row["area_px"])
            if area < min_area:
                continue
            particles.append({
                "particle_id": float(row["particle_id"]),
                "center_x_px": float(row["center_x_px"]),
                "center_y_px": float(row["center_y_px"]),
                "area_px": area,
            })
    return particles


def equivalent_radius(area: float, area_scale: float, radius_scale: float) -> float:
    """由面积计算等效圆半径，再用 radius_scale 控制烧结重叠程度。"""
    return math.sqrt(area * area_scale / math.pi) * radius_scale


def make_disks(
    particles: list[dict[str, float]],
    area_scale: float,
    radius_scale: float,
    resolution: int,
) -> list[Polygon]:
    """把每个颗粒参数化为一个圆盘；圆盘之间的重叠用于模拟烧结团聚。"""
    disks = []
    for particle in particles:
        radius = equivalent_radius(float(particle["area_px"]), area_scale, radius_scale)
        disk = Point(float(particle["center_x_px"]), float(particle["center_y_px"])).buffer(
            radius,
            resolution=resolution,
        )
        disks.append(disk)
    return disks


def geometry_parts(geom) -> list[Polygon]:
    """Shapely 的并集结果可能是 Polygon 或 MultiPolygon，统一拆成列表。"""
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return [part for part in getattr(geom, "geoms", []) if isinstance(part, Polygon)]


def smooth_necks(geom, neck_radius: float, simplify_tolerance: float):
    """Round particle necks after boolean union for COMSOL-friendly meshing."""
    if neck_radius > 0:
        geom = geom.buffer(neck_radius, join_style=1).buffer(-neck_radius, join_style=1)
    if simplify_tolerance > 0:
        geom = geom.simplify(simplify_tolerance, preserve_topology=True)
    return geom


def polygon_to_cv_points(poly: Polygon) -> np.ndarray:
    coords = np.asarray(poly.exterior.coords, dtype=np.float64)
    return np.round(coords).astype(np.int32).reshape((-1, 1, 2))


def draw_preview(
    image: np.ndarray,
    particles: list[dict[str, float]],
    disks: list[Polygon],
    union_parts: list[Polygon],
    out_prefix: Path,
    area_scale: float,
    radius_scale: float,
) -> None:
    """输出四张预览图：输入圆、并集填充图、并集叠加图、并集轮廓图。"""
    height, width = image.shape[:2]
    base = image.copy()
    if base.ndim == 2:
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    else:
        base = base.copy()

    disks_preview = base.copy()
    union_overlay = base.copy()
    union_outline = base.copy()
    white_union = np.full((height, width, 3), 255, dtype=np.uint8)

    for particle, disk in zip(particles, disks):
        radius = equivalent_radius(float(particle["area_px"]), area_scale, radius_scale)
        center = (int(round(float(particle["center_x_px"]))), int(round(float(particle["center_y_px"]))))
        cv2.circle(disks_preview, center, int(round(radius)), (200, 170, 80), 1, cv2.LINE_AA)
        cv2.drawMarker(disks_preview, center, (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)

    rng = np.random.default_rng(20260612)
    for part in union_parts:
        points = polygon_to_cv_points(part)
        color = tuple(int(v) for v in rng.integers(70, 235, size=3))
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [points], 255, lineType=cv2.LINE_AA)
        cv2.fillPoly(white_union, [points], color, lineType=cv2.LINE_AA)
        cv2.polylines(white_union, [points], True, (30, 30, 30), 2, lineType=cv2.LINE_AA)
        union_overlay[mask > 0] = (union_overlay[mask > 0] * 0.40 + np.array(color) * 0.60).astype(np.uint8)
        cv2.polylines(union_outline, [points], True, (0, 0, 255), 2, lineType=cv2.LINE_AA)

    save_png(out_prefix.with_name(out_prefix.name + "_input_circles.png"), disks_preview)
    save_png(out_prefix.with_name(out_prefix.name + "_union_filled.png"), white_union)
    save_png(out_prefix.with_name(out_prefix.name + "_union_overlay.png"), union_overlay)
    save_png(out_prefix.with_name(out_prefix.name + "_union_outlines.png"), union_outline)


def write_input_circles_dxf(
    path: Path,
    particles: list[dict[str, float]],
    height: int,
    area_scale: float,
    radius_scale: float,
    dxf_scale: float,
) -> None:
    """输出原始重叠圆。COMSOL 中也可以导入这个文件后自行做布尔并集。"""
    doc = ezdxf.new("R2010")
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()
    doc.layers.new("OVERLAPPING_CIRCLES", dxfattribs={"color": 4})
    for particle in particles:
        radius = equivalent_radius(float(particle["area_px"]), area_scale, radius_scale)
        msp.add_circle(
            (
                float(particle["center_x_px"]) * dxf_scale,
                (height - float(particle["center_y_px"])) * dxf_scale,
            ),
            radius=radius * dxf_scale,
            dxfattribs={"layer": "OVERLAPPING_CIRCLES"},
        )
    doc.saveas(path)


def write_union_dxf(path: Path, union_parts: list[Polygon], height: int, dxf_scale: float) -> None:
    """输出已经布尔并集后的团聚体边界。"""
    doc = ezdxf.new("R2010")
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()
    doc.layers.new("SINTERED_AGGLOMERATES", dxfattribs={"color": 7})
    for part in union_parts:
        exterior = [(float(x) * dxf_scale, float(height - y) * dxf_scale) for x, y in part.exterior.coords]
        msp.add_lwpolyline(exterior, close=True, dxfattribs={"layer": "SINTERED_AGGLOMERATES"})
        for interior in part.interiors:
            hole = [(float(x) * dxf_scale, float(height - y) * dxf_scale) for x, y in interior.coords]
            msp.add_lwpolyline(hole, close=True, dxfattribs={"layer": "SINTERED_AGGLOMERATES"})
    doc.saveas(path)


def write_agglomerate_csv(path: Path, union_parts: list[Polygon], dxf_scale: float) -> None:
    """保存每个并集团聚体的面积、等效直径和中心坐标。"""
    fields = [
        "agglomerate_id",
        "area_px",
        "equivalent_diameter_px",
        "center_x_px",
        "center_y_px",
        "area_scaled",
        "equivalent_diameter_scaled",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for idx, part in enumerate(union_parts, start=1):
            centroid = part.centroid
            area = float(part.area)
            diameter = 2.0 * math.sqrt(area / math.pi)
            writer.writerow({
                "agglomerate_id": idx,
                "area_px": round(area, 3),
                "equivalent_diameter_px": round(diameter, 3),
                "center_x_px": round(float(centroid.x), 3),
                "center_y_px": round(float(centroid.y), 3),
                "area_scaled": round(area * dxf_scale * dxf_scale, 6),
                "equivalent_diameter_scaled": round(diameter * dxf_scale, 6),
            })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="1_i311")
    parser.add_argument("--particles-csv", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "sintered_agglomerate_outputs"))
    # area-scale 改变颗粒面积；radius-scale 额外放大半径，是制造相邻圆盘重叠的主要参数。
    parser.add_argument("--area-scale", type=float, default=1.0)
    parser.add_argument("--radius-scale", type=float, default=1.08)
    # min-area 可删除很小颗粒，减少 COMSOL 几何和网格压力。
    parser.add_argument("--min-area", type=float, default=0.0)
    # resolution 控制 Shapely 圆盘离散精度；越大越圆，但并集 DXF 折线点越多。
    parser.add_argument("--resolution", type=int, default=24)
    # smooth-union 对并集边界做轻微圆角平滑，0 表示不额外处理。
    parser.add_argument("--smooth-union", type=float, default=0.0)
    # neck-radius 专门用于圆滑颗粒重叠后的颈部凹角；新任务建议优先用它。
    parser.add_argument("--neck-radius", type=float, default=0.0)
    # simplify-tolerance 用于减少并集 DXF 的短边数量，降低 COMSOL 网格压力。
    parser.add_argument("--simplify-tolerance", type=float, default=0.0)
    # dxf-scale 用于把像素坐标换算成 COMSOL 几何单位。
    parser.add_argument("--dxf-scale", type=float, default=1.0)
    args = parser.parse_args()

    particles_csv = Path(args.particles_csv) if args.particles_csv else ROOT / "particle_analysis" / f"{args.sample}_particles.csv"
    image_path = Path(args.image) if args.image else ROOT / "cellpose_dataset" / f"{args.sample}_sem.png"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    image = np.array(Image.open(image_path).convert("RGB"))
    height = image.shape[0]
    particles = read_particles(particles_csv, args.min_area)
    disks = make_disks(particles, args.area_scale, args.radius_scale, args.resolution)
    # 核心步骤：把所有重叠圆盘做布尔并集，得到烧结后的团聚体轮廓。
    union_geom = unary_union(disks)

    neck_radius = max(float(args.neck_radius), float(args.smooth_union))
    union_geom = smooth_necks(union_geom, neck_radius, float(args.simplify_tolerance))

    union_parts = sorted(geometry_parts(union_geom), key=lambda item: item.area, reverse=True)
    prefix = out_dir / f"{args.sample}_sintered_r{args.radius_scale:g}_a{args.area_scale:g}_neck{neck_radius:g}"

    draw_preview(image, particles, disks, union_parts, prefix, args.area_scale, args.radius_scale)
    write_input_circles_dxf(prefix.with_name(prefix.name + "_input_circles.dxf"), particles, height, args.area_scale, args.radius_scale, args.dxf_scale)
    write_union_dxf(prefix.with_name(prefix.name + "_boolean_union.dxf"), union_parts, height, args.dxf_scale)
    write_agglomerate_csv(prefix.with_name(prefix.name + "_agglomerates.csv"), union_parts, args.dxf_scale)

    print(f"input_particles={len(particles)}")
    print(f"agglomerates_after_union={len(union_parts)}")
    print(f"radius_scale={args.radius_scale}")
    print(f"area_scale={args.area_scale}")
    print(f"neck_radius={neck_radius}")
    print(f"simplify_tolerance={args.simplify_tolerance}")
    print(f"saved prefix: {prefix}")


if __name__ == "__main__":
    main()
