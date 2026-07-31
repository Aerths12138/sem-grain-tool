import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

try:
    import ezdxf
except ImportError:  # DXF output is optional.
    ezdxf = None


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CircleShape:
    center_x: float
    center_y: float
    radius: float

    @property
    def area(self) -> float:
        return math.pi * self.radius * self.radius


def circle_overlap_stats(circles: list[CircleShape], gap: float, tolerance: float = 0.05) -> tuple[int, float]:
    overlap_count = 0
    max_overlap = 0.0
    for i in range(len(circles)):
        a = circles[i]
        for j in range(i + 1, len(circles)):
            b = circles[j]
            distance = math.hypot(a.center_x - b.center_x, a.center_y - b.center_y)
            overlap = a.radius + b.radius + gap - distance
            if overlap > tolerance:
                overlap_count += 1
                max_overlap = max(max_overlap, overlap)
    return overlap_count, max_overlap


def resolve_circle_overlaps(
    circles: list[CircleShape],
    original_centers: np.ndarray,
    max_displacement_ratio: float,
    gap: float,
    iterations: int,
    damping: float,
    seed: int,
) -> list[CircleShape]:
    """Move circle centers to reduce overlap while preserving each circle area.

    The displacement of each generated center is constrained so the original
    particle center remains inside its generated circle.
    """
    if not circles:
        return circles

    rng = np.random.default_rng(seed)
    centers = np.array([[circle.center_x, circle.center_y] for circle in circles], dtype=np.float64)
    radii = np.array([circle.radius for circle in circles], dtype=np.float64)
    max_displacements = np.maximum(0.0, radii * max_displacement_ratio)

    for _ in range(max(0, iterations)):
        delta = np.zeros_like(centers)
        max_step_overlap = 0.0
        for i in range(len(circles)):
            for j in range(i + 1, len(circles)):
                vector = centers[j] - centers[i]
                distance = float(np.linalg.norm(vector))
                if distance < 1e-9:
                    angle = rng.uniform(0, 2 * math.pi)
                    direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float64)
                    distance = 1e-9
                else:
                    direction = vector / distance
                overlap = radii[i] + radii[j] + gap - distance
                if overlap <= 0:
                    continue

                max_step_overlap = max(max_step_overlap, overlap)
                weight_i = radii[j] / (radii[i] + radii[j])
                weight_j = radii[i] / (radii[i] + radii[j])
                delta[i] -= direction * overlap * weight_i
                delta[j] += direction * overlap * weight_j

        centers += delta * damping

        offsets = centers - original_centers
        distances = np.linalg.norm(offsets, axis=1)
        too_far = distances > max_displacements
        if np.any(too_far):
            scale = np.ones_like(distances)
            scale[too_far] = max_displacements[too_far] / distances[too_far]
            centers = original_centers + offsets * scale[:, None]

        if max_step_overlap < 0.01:
            break

    return [
        CircleShape(float(center[0]), float(center[1]), float(radius))
        for center, radius in zip(centers, radii)
    ]


def save_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"Could not encode PNG: {path}")
    encoded.tofile(str(path))


def read_particles(path: Path) -> list[dict[str, float]]:
    particles: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            particles.append({
                "particle_id": float(row["particle_id"]),
                "center_x_px": float(row["center_x_px"]),
                "center_y_px": float(row["center_y_px"]),
                "area_px": float(row["area_px"]),
                "equivalent_diameter_px": float(row["equivalent_diameter_px"]),
            })
    return particles


def polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def chaikin_closed(points: np.ndarray, iterations: int) -> np.ndarray:
    out = np.asarray(points, dtype=np.float64)
    for _ in range(iterations):
        nxt = np.roll(out, -1, axis=0)
        q = 0.75 * out + 0.25 * nxt
        r = 0.25 * out + 0.75 * nxt
        smoothed = np.empty((len(out) * 2, 2), dtype=np.float64)
        smoothed[0::2] = q
        smoothed[1::2] = r
        out = smoothed
    return out


def rescale_to_area(points: np.ndarray, center: np.ndarray, target_area: float) -> np.ndarray:
    current_area = polygon_area(points)
    if current_area <= 0 or target_area <= 0:
        return points
    scale = math.sqrt(target_area / current_area)
    return center + (points - center) * scale


def generate_particle_shape(
    center_x: float,
    center_y: float,
    area: float,
    rng: np.random.Generator,
    vertices: int,
    radius_jitter: float,
    angle_jitter: float,
    smooth_iterations: int,
) -> np.ndarray:
    center = np.array([center_x, center_y], dtype=np.float64)
    base_radius = math.sqrt(area / math.pi)
    base_angles = np.linspace(0, 2 * math.pi, vertices, endpoint=False)
    angle_offsets = rng.uniform(-angle_jitter, angle_jitter, size=vertices) * (2 * math.pi / vertices)
    angles = np.sort(base_angles + angle_offsets)
    radii = base_radius * rng.uniform(1.0 - radius_jitter, 1.0 + radius_jitter, size=vertices)
    points = np.column_stack((center_x + np.cos(angles) * radii, center_y + np.sin(angles) * radii))
    points = rescale_to_area(points, center, area)
    if smooth_iterations > 0:
        points = chaikin_closed(points, smooth_iterations)
        points = rescale_to_area(points, center, area)
    return points


def clip_points_to_canvas(points: np.ndarray, width: int, height: int) -> np.ndarray:
    clipped = points.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, width - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, height - 1)
    return clipped


def circle_to_polygon(circle: CircleShape, vertices: int = 96) -> np.ndarray:
    angles = np.linspace(0, 2 * math.pi, vertices, endpoint=False)
    return np.column_stack((
        circle.center_x + np.cos(angles) * circle.radius,
        circle.center_y + np.sin(angles) * circle.radius,
    ))


def draw_outputs(
    image: np.ndarray,
    shapes: list[np.ndarray | CircleShape],
    centers: list[tuple[int, int, int]],
    output_prefix: Path,
) -> None:
    height, width = image.shape[:2]
    white = np.full((height, width, 3), 255, dtype=np.uint8)
    overlay = image.copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    else:
        overlay = overlay.copy()
    outlines = overlay.copy()

    rng = np.random.default_rng(20260612)
    for shape in shapes:
        color = tuple(int(v) for v in rng.integers(70, 235, size=3))
        mask = np.zeros((height, width), dtype=np.uint8)
        if isinstance(shape, CircleShape):
            center = (int(round(shape.center_x)), int(round(shape.center_y)))
            radius = max(1, int(round(shape.radius)))
            cv2.circle(white, center, radius, color, thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(white, center, radius, (35, 35, 35), thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(mask, center, radius, 255, thickness=-1, lineType=cv2.LINE_AA)
            overlay[mask > 0] = (overlay[mask > 0] * 0.42 + np.array(color) * 0.58).astype(np.uint8)
            cv2.circle(outlines, center, radius, (0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
        else:
            points = np.round(shape).astype(np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(white, [points], color, lineType=cv2.LINE_AA)
            cv2.polylines(white, [points], True, (35, 35, 35), 2, lineType=cv2.LINE_AA)
            cv2.fillPoly(mask, [points], 255, lineType=cv2.LINE_AA)
            overlay[mask > 0] = (overlay[mask > 0] * 0.42 + np.array(color) * 0.58).astype(np.uint8)
            cv2.polylines(outlines, [points], True, (0, 0, 255), 2, lineType=cv2.LINE_AA)

    centers_preview = overlay.copy()
    for particle_id, x, y in centers:
        cv2.drawMarker(centers_preview, (x, y), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)
        cv2.putText(centers_preview, str(particle_id), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
        cv2.putText(centers_preview, str(particle_id), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    save_png(output_prefix.with_name(output_prefix.name + "_filled.png"), white)
    save_png(output_prefix.with_name(output_prefix.name + "_overlay.png"), overlay)
    save_png(output_prefix.with_name(output_prefix.name + "_outlines.png"), outlines)
    save_png(output_prefix.with_name(output_prefix.name + "_centers.png"), centers_preview)


def shape_area(shape: np.ndarray | CircleShape) -> float:
    if isinstance(shape, CircleShape):
        return shape.area
    return polygon_area(shape)


def shape_vertex_count(shape: np.ndarray | CircleShape) -> int:
    if isinstance(shape, CircleShape):
        return 0
    return int(len(shape))


def write_shape_csv(path: Path, particles: list[dict[str, float]], shapes: list[np.ndarray | CircleShape], area_scale: float) -> None:
    fieldnames = [
        "particle_id",
        "center_x_px",
        "center_y_px",
        "generated_center_x_px",
        "generated_center_y_px",
        "center_shift_px",
        "source_area_px",
        "target_area_px",
        "generated_area_px",
        "equivalent_diameter_px",
        "vertex_count",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for particle, shape in zip(particles, shapes):
            source_area = float(particle["area_px"])
            target_area = source_area * area_scale
            if isinstance(shape, CircleShape):
                generated_center_x = shape.center_x
                generated_center_y = shape.center_y
            else:
                generated_center_x = float(particle["center_x_px"])
                generated_center_y = float(particle["center_y_px"])
            center_shift = math.hypot(
                generated_center_x - float(particle["center_x_px"]),
                generated_center_y - float(particle["center_y_px"]),
            )
            writer.writerow({
                "particle_id": int(particle["particle_id"]),
                "center_x_px": round(float(particle["center_x_px"]), 3),
                "center_y_px": round(float(particle["center_y_px"]), 3),
                "generated_center_x_px": round(generated_center_x, 3),
                "generated_center_y_px": round(generated_center_y, 3),
                "center_shift_px": round(center_shift, 3),
                "source_area_px": round(source_area, 3),
                "target_area_px": round(target_area, 3),
                "generated_area_px": round(shape_area(shape), 3),
                "equivalent_diameter_px": round(2.0 * math.sqrt(target_area / math.pi), 3),
                "vertex_count": shape_vertex_count(shape),
            })


def write_dxf(path: Path, shapes: list[np.ndarray | CircleShape], height: int, scale: float) -> None:
    if ezdxf is None:
        print("ezdxf is not installed; skipped DXF output")
        return
    doc = ezdxf.new("R2010")
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()
    doc.layers.new("PARTICLE_SHAPES", dxfattribs={"color": 7})
    for shape in shapes:
        if isinstance(shape, CircleShape):
            msp.add_circle(
                (shape.center_x * scale, (height - shape.center_y) * scale),
                radius=shape.radius * scale,
                dxfattribs={"layer": "PARTICLE_SHAPES"},
            )
        else:
            points = [(float(x) * scale, float(height - y) * scale) for x, y in shape]
            msp.add_lwpolyline(points, close=True, dxfattribs={"layer": "PARTICLE_SHAPES"})
    doc.saveas(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", default="1_i313")
    parser.add_argument("--particles-csv", default=None)
    parser.add_argument("--image", default=None)
    parser.add_argument("--output-dir", default=str(ROOT / "coordinate_shape_outputs"))
    parser.add_argument("--mode", choices=["circle", "smooth", "polygon"], default="circle")
    parser.add_argument("--area-scale", type=float, default=1.0)
    parser.add_argument("--vertices", type=int, default=14)
    parser.add_argument("--radius-jitter", type=float, default=0.18)
    parser.add_argument("--angle-jitter", type=float, default=0.35)
    parser.add_argument("--smooth-iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--clip-to-canvas", action="store_true")
    parser.add_argument("--dxf-scale", type=float, default=1.0)
    parser.add_argument("--allow-overlap", action="store_true")
    parser.add_argument("--circle-gap", type=float, default=0.0)
    parser.add_argument("--resolve-iterations", type=int, default=800)
    parser.add_argument("--resolve-damping", type=float, default=0.25)
    parser.add_argument("--max-center-shift-ratio", type=float, default=1.0)
    args = parser.parse_args()

    particles_csv = Path(args.particles_csv) if args.particles_csv else ROOT / "particle_analysis" / f"{args.sample}_particles.csv"
    image_path = Path(args.image) if args.image else ROOT / "cellpose_dataset" / f"{args.sample}_sem.png"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    image = np.array(Image.open(image_path).convert("RGB"))
    height, width = image.shape[:2]
    particles = read_particles(particles_csv)

    smooth_iterations = 0 if args.mode == "polygon" else args.smooth_iterations
    rng = np.random.default_rng(args.seed)

    shapes: list[np.ndarray | CircleShape] = []
    original_centers: list[tuple[float, float]] = []
    centers: list[tuple[int, int, int]] = []
    for particle in particles:
        target_area = float(particle["area_px"]) * args.area_scale
        center_x = float(particle["center_x_px"])
        center_y = float(particle["center_y_px"])
        original_centers.append((center_x, center_y))
        if args.mode == "circle":
            shape = CircleShape(center_x, center_y, math.sqrt(target_area / math.pi))
        else:
            shape = generate_particle_shape(
                center_x,
                center_y,
                target_area,
                rng,
                max(3, int(args.vertices)),
                max(0.0, float(args.radius_jitter)),
                max(0.0, float(args.angle_jitter)),
                max(0, int(smooth_iterations)),
            )
        if args.clip_to_canvas and not isinstance(shape, CircleShape):
            shape = clip_points_to_canvas(shape, width, height)
        shapes.append(shape)
        centers.append((
            int(particle["particle_id"]),
            int(round(float(particle["center_x_px"]))),
            int(round(float(particle["center_y_px"]))),
        ))

    overlap_before = None
    overlap_after = None
    if args.mode == "circle":
        circle_shapes = [shape for shape in shapes if isinstance(shape, CircleShape)]
        overlap_before = circle_overlap_stats(circle_shapes, args.circle_gap)
        if not args.allow_overlap:
            shapes = resolve_circle_overlaps(
                circle_shapes,
                np.array(original_centers, dtype=np.float64),
                max(0.0, float(args.max_center_shift_ratio)),
                float(args.circle_gap),
                int(args.resolve_iterations),
                float(args.resolve_damping),
                int(args.seed),
            )
            overlap_after = circle_overlap_stats([shape for shape in shapes if isinstance(shape, CircleShape)], args.circle_gap)

    suffix = "overlap" if args.allow_overlap or args.mode != "circle" else "nonoverlap"
    prefix = output_dir / f"{args.sample}_{args.mode}_{suffix}_a{args.area_scale:g}_gap{args.circle_gap:g}_area_from_centers"
    draw_outputs(image, shapes, centers, prefix)
    write_shape_csv(prefix.with_name(prefix.name + "_data.csv"), particles, shapes, args.area_scale)
    write_dxf(prefix.with_name(prefix.name + ".dxf"), shapes, height, args.dxf_scale)

    print(f"particles={len(particles)}")
    print(f"mode={args.mode}")
    print(f"area_scale={args.area_scale}")
    if overlap_before is not None:
        print(f"overlaps_before={overlap_before[0]}, max_overlap_before={overlap_before[1]:.3f}")
    if overlap_after is not None:
        print(f"overlaps_after={overlap_after[0]}, max_overlap_after={overlap_after[1]:.3f}")
    print(f"saved prefix: {prefix}")


if __name__ == "__main__":
    main()
