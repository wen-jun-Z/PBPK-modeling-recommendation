"""Three native SVG plot schematics following the selected reference artwork.

Coordinates are global in the original 1605 × 980 canvas.  Paths, marker
positions and colors follow the reference; no raster background is carried
over.  These are explanatory schematics, not additional experimental results.
"""

from __future__ import annotations

from xml.etree import ElementTree as ET

from english_workflow_art import MATH_DIR, _mathtext_backend

_mathtext_backend()
from matplotlib import rc_context
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MPath
from matplotlib.textpath import TextPath


NS = "http://www.w3.org/2000/svg"
INK = "#101315"
AXIS = "#202D66"
GREEN = "#176A53"
PURPLE = "#5D2698"
GOLD = "#A56A0C"


def _el(parent: ET.Element, tag: str, **attrs) -> ET.Element:
    return ET.SubElement(parent, f"{{{NS}}}{tag}", {
        key.replace("_", "-"): str(value) for key, value in attrs.items()
    })


def _path(parent: ET.Element, d: str, color: str, width: float = 2,
          fill: str = "none", **attrs) -> ET.Element:
    return _el(parent, "path", d=d, fill=fill, stroke=color,
               stroke_width=width, stroke_linecap="round",
               stroke_linejoin="round", **attrs)


def _text(parent: ET.Element, x: float, y: float, value: str,
          size: float = 18, color: str = INK, bold: bool = False,
          align: str = "left", vertical: bool = False,
          math: bool = False) -> ET.Element:
    """Vector text in the same condensed DejaVu/STIX style as the main figure.

    Horizontal y denotes the visible text top; vertical (x,y) is its center.
    """
    font = "STIXGeneral.ttf" if math else (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    )
    with rc_context({"mathtext.fontset": "stix"}):
        shape = TextPath((0, 0), value, size=size,
                         prop=FontProperties(fname=str(MATH_DIR / font)))
    box = shape.get_extents()
    sx = 1.0 if math else .84
    commands = []
    codes = {MPath.MOVETO: "M", MPath.LINETO: "L", MPath.CURVE3: "Q",
             MPath.CURVE4: "C", MPath.CLOSEPOLY: "Z"}
    for vertices, code in shape.iter_segments(curves=True, simplify=False):
        op = codes[code]
        commands.append(op + (" ".join(f"{v:.4f}" for v in vertices)
                               if op != "Z" else ""))
    if vertical:
        xx = -(box.x0 + box.x1) * sx / 2
        yy = (box.y0 + box.y1) / 2
        transform = f"translate({x:g} {y:g}) rotate(-90) translate({xx:g} {yy:g}) scale({sx:g} -1)"
    else:
        xx = x - box.x0 * sx
        if align == "center":
            xx -= box.width * sx / 2
        elif align == "right":
            xx -= box.width * sx
        transform = f"translate({xx:g} {y + box.y1:g}) scale({sx:g} -1)"
    return _el(parent, "path", d=" ".join(commands), fill=color,
               transform=transform, aria_label=value)


def _axes(group: ET.Element, left: float, top: float, right: float,
          bottom: float) -> None:
    _path(group, f"M{left:g} {top:g}V{bottom:g}H{right:g}", AXIS, 2.3)


def _development(group: ET.Element) -> None:
    # Original top-right development trace: steep decline and a shallow tail.
    _axes(group, 998, 94, 1291, 275)
    upper = "M1015 112 C1044 154 1071 187 1090 197 C1130 219 1170 231 1283 232"
    lower = "M1015 128 C1046 172 1078 207 1107 224 C1151 250 1215 267 1283 271"
    for d in (upper, lower):
        _path(group, d, GREEN, 1.25, stroke_dasharray="5 4", stroke_opacity=".36")
    curve = ("M1015 120 C1034 147 1053 169 1082 192 "
             "C1111 215 1137 225 1170 236 "
             "C1205 249 1242 255 1283 253")
    _path(group, curve, "#FFFFFF", 5)
    _path(group, curve, GREEN, 2.7)
    for x, y in ((1015, 118), (1051, 157), (1089, 205),
                 (1142, 221), (1204, 252), (1282, 247)):
        _el(group, "circle", cx=x, cy=y, r=6.0, fill="#FFFFFF",
            stroke=GREEN, stroke_width=2)
    _text(group, 973, 205, "Concentration", 17.5, vertical=True)
    _text(group, 1144, 287, "Time", 18, bold=True, align="center")


def _mass_bound(group: ET.Element) -> None:
    # The screenshot has an unfilled purple trajectory below the upper bound.
    # Leaving the region unfilled removes the screenshot's baked checkerboard.
    _axes(group, 239, 673, 435, 790)
    _path(group, "M241 702H433", GOLD, 2.4, stroke_dasharray="7 5")
    _text(group, 349, 678, "Upper bound", 18, GOLD, bold=True)
    curve = ("M242 788 C253 761 262 725 276 717 "
             "C285 711 292 719 301 731 "
             "C316 749 330 760 350 768 "
             "C376 778 406 782 431 783")
    _path(group, curve, "#FFFFFF", 5.2)
    _path(group, curve, PURPLE, 3.0)
    _el(group, "circle", cx=282, cy=716, r=5.7, fill=PURPLE,
        stroke="#FFFFFF", stroke_width=1.35)
    _text(group, 217, 744, r"$C_p(t)$", 22, AXIS, vertical=True, math=True)
    _text(group, 335, 804, "Time", 18, bold=True, align="center")
    # Arrow from the observed curve to the adjacent feasibility decision.
    _path(group, "M440 722H450", GOLD, 3)
    _el(group, "path", d="M449 716L459 722L449 728Z", fill=GOLD)


def _target(group: ET.Element) -> None:
    curve = ("M1303 748 C1314 711 1324 656 1337 655 "
             "C1350 653 1357 678 1376 695 "
             "C1398 716 1421 730 1451 739")
    area = curve + "L1451 750H1303Z"
    defs = _el(group, "defs")
    clip = _el(defs, "clipPath", id="reference_target_auc_clip",
               clipPathUnits="userSpaceOnUse")
    _el(clip, "path", d=area)
    _el(group, "path", d=area, fill="#E8D9F6", fill_opacity=".8")
    # Fine diagonal hatching retains the original AUC-area appearance.
    hatch = _el(group, "g", clip_path="url(#reference_target_auc_clip)")
    for x in range(1190, 1455, 4):
        _path(hatch, f"M{x} 754L{x + 120} 634", "#C3A7DE", .6,
              stroke_opacity=".7")
    _axes(group, 1300, 625, 1453, 751)
    _path(group, curve, "#FFFFFF", 5.3)
    _path(group, curve, PURPLE, 3)
    _el(group, "circle", cx=1337, cy=655, r=7.0, fill=PURPLE,
        stroke="#FFFFFF", stroke_width=1.3)
    _text(group, 1350, 638, "Cmax", 21, PURPLE, bold=True)
    _text(group, 1340, 715, "AUC", 20.5, PURPLE, bold=True)
    _text(group, 1282, 691, "Concentration", 17.5, vertical=True)
    _text(group, 1375, 762, "Time", 18, bold=True, align="center")


def reference_plot(name: str) -> ET.Element:
    """Return dev_curve, mass_curve or target_curve as a global SVG group."""
    renderers = {"dev_curve": _development, "mass_curve": _mass_bound,
                 "target_curve": _target}
    if name not in renderers:
        raise ValueError(f"Unknown reference plot {name!r}")
    group = ET.Element(f"{{{NS}}}g", {"id": "ref_" + name})
    title = _el(group, "title")
    title.text = {
        "dev_curve": "Development calibration schematic with observed samples",
        "mass_curve": "Visible concentration and a mass-conservation upper bound",
        "target_curve": "Locked-model target trajectory, peak and AUC schematic",
    }[name]
    renderers[name](group)
    return group
