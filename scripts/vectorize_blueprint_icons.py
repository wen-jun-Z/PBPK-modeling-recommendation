#!/usr/bin/env python3
"""Trace the chosen PBPK blueprint's illustrations into reusable SVG paths.

The source raster is used only during this build. The generated atlas has no
embedded or linked bitmap assets, labels, math, or workflow connectors.

Run with runtime/app-env/bin/python scripts/vectorize_blueprint_icons.py.
vtracer is installed locally in runtime/figure_tools, not system-wide.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime/figure_tools"))

import cairosvg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from scipy.interpolate import splprep, splev
import vtracer


SOURCE = ROOT / "results/figures/workflow/figure_2_blueprint_original_transparent.png"
OUTPUT = ROOT / "results/figures/workflow/blueprint_original_icons.svg"
REVIEW = ROOT / "runtime/presentation_build/blueprint_vector_review"
NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", NS)

# Coordinates correspond to the original 1605 x 980 illustration.
BOXES = {
    "human": (38, 82, 188, 295),
    "evidence": (221, 122, 375, 282),
    "liver": (690, 102, 771, 160),
    "kidneys": (692, 172, 768, 244),
    "balance_and_bottle": (41, 658, 184, 774),
    "locked_model": (912, 615, 1052, 742),
    "target_human_iv": (1115, 602, 1203, 700),
    "report": (1481, 627, 1569, 736),
    "audit_clipboard": (1505, 820, 1574, 911),
    "lock_small": (634, 636, 684, 694),
    "wrench": (762, 633, 837, 689),
}


def prepare_icon(image: Image.Image, name: str) -> Image.Image:
    """Remove neighboring diagram content and preserve illustration colors."""
    rgba = np.array(image.crop(BOXES[name]).convert("RGBA"))
    rgb = rgba[:, :, :3].astype(np.int16)
    alpha = rgba[:, :, 3].copy()
    yy, xx = np.indices(alpha.shape)

    if name == "human":
        alpha[(xx >= 143) & (yy < 146)] = 0  # neighboring plus sign
    elif name == "balance_and_bottle":
        alpha[(xx >= 106) & (yy >= 69) & (yy <= 89)] = 0
    elif name == "locked_model":
        # Follow the tilted sheet's lower edge to exclude the label beneath it.
        alpha[yy > 125 - 0.116 * xx] = 0
    elif name == "target_human_iv":
        alpha[yy >= 94] = 0
        # A gray checkerboard remnant lies outside the blue head contour.
        neutral = (rgb.max(axis=2) - rgb.min(axis=2) < 25)
        neutral &= (rgb.max(axis=2) < 235) & (rgb.min(axis=2) > 100)
        alpha[neutral & ((xx < 23) | (xx > 31)) & (yy < 34)] = 0
    elif name == "audit_clipboard":
        alpha[yy < 8] = 0  # tip of the incoming flow arrow
    elif name in {"lock_small", "wrench"}:
        # These two drawings sit on a filled panel. Their pale neutral panel
        # pixels are not part of their warm lock / cool blue wrench colors.
        neutral = (rgb.max(axis=2) - rgb.min(axis=2) < 30)
        neutral &= rgb.min(axis=2) > 180
        alpha[neutral] = 0

    # Do not promote faint extraction noise to opaque vector contours.
    alpha = np.where(alpha >= 200, 255, 0).astype(np.uint8)
    # Restore only tiny enclosed holes introduced by checkerboard extraction.
    # The input retains the original RGB values even at transparent pixels.
    holes = ndimage.binary_fill_holes(alpha > 0) & (alpha == 0)
    hole_labels, hole_count = ndimage.label(holes)
    if hole_count:
        hole_areas = np.bincount(hole_labels.ravel())
        alpha[(hole_labels > 0) & (hole_areas[hole_labels] <= 12)] = 255
    labels, count = ndimage.label(alpha > 0)
    if count:
        areas = np.bincount(labels.ravel())
        alpha[(labels > 0) & (areas[labels] < 5)] = 0
    rgba[:, :, 3] = alpha
    return Image.fromarray(rgba)


def smooth_path(data: str, tolerance: float = 0.7) -> str:
    """Fit cubic splines with a subpixel smoothing target.

    VTracer keeps tiny staircase corners at transparent boundaries. A periodic
    cubic fit removes those raster-grid corners while preserving each contour,
    its direction, its color, and the trace's stacking order.
    """
    output = []
    for chunk in re.findall(r"M[^M]*", data):
        tokens = re.findall(r"[MCZL]|[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", chunk)
        points = []
        index = 0
        point = start = None
        while index < len(tokens):
            command = tokens[index]
            index += 1
            if command in {"M", "L"}:
                point = np.array([float(tokens[index]), float(tokens[index + 1])])
                index += 2
                points.append(point)
                if command == "M":
                    start = point.copy()
            elif command == "C":
                control1 = np.array([float(tokens[index]), float(tokens[index + 1])])
                control2 = np.array([float(tokens[index + 2]), float(tokens[index + 3])])
                end = np.array([float(tokens[index + 4]), float(tokens[index + 5])])
                index += 6
                samples = max(2, min(40, math.ceil(np.linalg.norm(end - point) / 1.5)))
                for t in np.linspace(0, 1, samples + 1)[1:]:
                    points.append(
                        (1 - t) ** 3 * point + 3 * (1 - t) ** 2 * t * control1
                        + 3 * (1 - t) * t * t * control2 + t ** 3 * end
                    )
                point = end
            elif command == "Z":
                if np.linalg.norm(points[-1] - start) > 1e-7:
                    points.append(start)
            else:
                raise ValueError(f"Unexpected VTracer command {command}")
        if len(points) < 12:
            output.append(chunk)
            continue
        points = np.array(points)
        distinct = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-7]
        points = points[distinct]
        try:
            spline, _ = splprep(points.T, s=len(points) * tolerance ** 2, per=True)
        except ValueError:
            output.append(chunk)
            continue
        knots = sorted({0., 1., *(float(t) for t in spline[0] if 0 < t < 1)})

        def coordinates(values: np.ndarray) -> str:
            return " ".join(f"{value:.3f}".rstrip("0").rstrip(".") or "0" for value in values)

        fitted = "M" + coordinates(splev(0, spline))
        for a, b in zip(knots[:-1], knots[1:]):
            start = np.array(splev(a, spline))
            end = np.array(splev(b, spline))
            velocity0 = np.array(splev(a, spline, der=1))
            velocity1 = np.array(splev(b, spline, der=1))
            dt = (b - a) / 3
            fitted += (
                " C" + coordinates(start + velocity0 * dt)
                + " " + coordinates(end - velocity1 * dt) + " " + coordinates(end)
            )
        output.append(fitted + " Z")
    return " ".join(output)


def main() -> None:
    REVIEW.mkdir(parents=True, exist_ok=True)
    original = Image.open(SOURCE).convert("RGBA")
    if original.size != (1605, 980):
        raise ValueError(f"Unexpected source dimensions: {original.size}")
    atlas = ET.Element(f"{{{NS}}}svg", {"version": "1.1"})
    ET.SubElement(atlas, f"{{{NS}}}title").text = (
        "Original PBPK blueprint illustrations, traced as vector paths"
    )
    defs = ET.SubElement(atlas, f"{{{NS}}}defs")
    stats = {}
    previews = []

    for name, box in BOXES.items():
        crop = prepare_icon(original, name)
        width, height = crop.size
        crop_path = REVIEW / f"{name}_prepared.png"
        trace_path = REVIEW / f"{name}_traced.svg"
        # A 2x antialiased sampling grid gives smooth subpixel spline outlines.
        enlarged = np.array(crop.resize((width * 2, height * 2), Image.Resampling.LANCZOS))
        enlarged[:, :, 3] = np.where(enlarged[:, :, 3] >= 128, 255, 0)
        Image.fromarray(enlarged).save(crop_path)
        vtracer.convert_image_to_svg_py(
            str(crop_path), str(trace_path),
            colormode="color", hierarchical="stacked", mode="spline",
            filter_speckle=3, color_precision=6, layer_difference=10,
            corner_threshold=60, length_threshold=3.5, max_iterations=10,
            splice_threshold=45, path_precision=3,
        )
        traced = ET.parse(trace_path).getroot()
        symbol = ET.SubElement(defs, f"{{{NS}}}symbol", {
            "id": name, "viewBox": f"0 0 {width} {height}",
        })
        for path in traced:
            if path.tag != f"{{{NS}}}path":
                continue
            path = copy.deepcopy(path)
            path.set("d", smooth_path(path.get("d")))
            # Each symbol contains paths only. Preserve the trace's local
            # transforms and convert its coordinates back to original units.
            path.set("transform", "scale(0.5) " + path.get("transform", ""))
            symbol.append(path)
        stats[name] = {"viewBox": [0, 0, width, height], "paths": len(symbol)}

        preview_svg = ET.Element(f"{{{NS}}}svg", {
            "width": str(width), "height": str(height),
            "viewBox": f"0 0 {width} {height}",
        })
        for path in symbol:
            preview_svg.append(copy.deepcopy(path))
        preview_png = REVIEW / f"{name}_vector.png"
        cairosvg.svg2png(
            bytestring=ET.tostring(preview_svg), write_to=str(preview_png),
            output_width=width * 3, output_height=height * 3,
        )
        previews.append((name, preview_png))

    ET.indent(atlas, space="  ")
    ET.ElementTree(atlas).write(OUTPUT, encoding="utf-8", xml_declaration=True)
    # An atlas is deliberately invisible on its own; contact sheets make each
    # symbol reviewable while keeping all QA raster files in the runtime tree.
    cell_w, cell_h, columns = 410, 490, 4
    sheet = Image.new("RGB", (cell_w * columns, cell_h * 3), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    for index, (name, path) in enumerate(previews):
        col, row = index % columns, index // columns
        x, y = col * cell_w, row * cell_h
        draw.text((x + 18, y + 15), name, fill="#222222", font=font)
        preview = Image.open(path).convert("RGBA")
        preview.thumbnail((cell_w - 36, cell_h - 68), Image.Resampling.LANCZOS)
        sheet.paste(preview, (x + 18, y + 54), preview)
    sheet.save(REVIEW / "vector_contact.png")
    (REVIEW / "trace_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    assert not atlas.findall(f".//{{{NS}}}image"), "Bitmap element in vector atlas"
    assert len(defs) == len(BOXES)
    print(json.dumps({
        "atlas": str(OUTPUT), "contact_sheet": str(REVIEW / "vector_contact.png"),
        "symbols": len(defs), "paths": sum(x["paths"] for x in stats.values()),
        "bytes": OUTPUT.stat().st_size,
    }, indent=2))


if __name__ == "__main__":
    main()
