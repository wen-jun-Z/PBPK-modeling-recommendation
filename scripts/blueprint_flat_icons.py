"""Editable flat scientific icons for the 1605 × 980 blueprint workflow.

``flat_icon(name)`` returns a fresh SVG group in the original figure's global
coordinates. These authored paths replace the seven former shaded artworks;
no images, gradients, filters, tracing data, fonts or external resources occur
inside the returned groups. The existing algorithm/layout remains unchanged.

Run this module to export a labelled contact sheet for visual inspection.
"""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

NS = "http://www.w3.org/2000/svg"
INK = "#243642"
BLUE = "#78A8BD"
GREEN = "#A8CCBB"
GOLD = "#E7D18E"
PURPLE = "#B7A6CA"
CORAL = "#EAA99F"
WHITE = "#FFFFFF"
PALE = "#EEF5F7"
OUTLINE = 3.6
DETAIL = 2.0

# Original canvas placement and local view boxes (x, y, width, height).
ICON_BOUNDS = {
    "human": (38.0, 83.0, 153.9, 210.9),
    "evidence": (220.0, 122.0, 154.0, 161.0),
    "drug_scales": (35.0, 657.0, 159.0, 115.0),
    "blueprint": (912.0, 620.0, 142.0, 122.0),
    "target_human": (1115.0, 604.0, 85.0, 90.0),
    "report": (1484.0, 630.0, 82.0, 103.0),
    "score": (1507.0, 826.0, 59.0, 80.0),
}


def _el(parent: ET.Element | None, tag: str, **attrs) -> ET.Element:
    attrs = {k.replace("_", "-"): str(v) for k, v in attrs.items()}
    return (ET.Element(f"{{{NS}}}{tag}", attrs) if parent is None
            else ET.SubElement(parent, f"{{{NS}}}{tag}", attrs))


def _path(g, d, fill="none", stroke=INK, width=OUTLINE, **attrs):
    return _el(g, "path", d=d, fill=fill, stroke=stroke,
               stroke_width=width, **attrs)


def _line(g, x1, y1, x2, y2, color=INK, width=DETAIL, **attrs):
    return _el(g, "line", x1=x1, y1=y1, x2=x2, y2=y2,
               stroke=color, stroke_width=width, **attrs)


def _rect(g, x, y, w, h, fill=WHITE, r=3, width=OUTLINE, **attrs):
    return _el(g, "rect", x=x, y=y, width=w, height=h, rx=r,
               fill=fill, stroke=INK, stroke_width=width, **attrs)


def _circle(g, x, y, r, fill, width=OUTLINE, **attrs):
    return _el(g, "circle", cx=x, cy=y, r=r, fill=fill,
               stroke=INK, stroke_width=width, **attrs)


def _group(name, title, transform=None):
    x, y, *_ = ICON_BOUNDS[name]
    g = _el(None, "g", id=f"flat_blueprint_{name}",
            transform=transform or f"translate({x:g} {y:g})",
            stroke_linecap="round", stroke_linejoin="round",
            aria_label=title, data_icon=name)
    _el(g, "title").text = title
    return g


def _heart(g, x, y, scale=1):
    h = _el(g, "g", transform=f"translate({x} {y}) scale({scale})")
    _path(h, "M 12 5 C 6 -3 -2 3 1 11 C 3 18 11 24 12 25 "
          "C 15 22 23 17 24 11 C 27 3 19 -3 12 5 Z", CORAL,
          width=3.0)
    return h


def _liver(g, x, y, scale=1):
    h = _el(g, "g", transform=f"translate({x} {y}) scale({scale})")
    _path(h, "M 2 8 C 11 -2 25 0 42 4 Q 51 5 51 10 "
          "C 46 17 37 20 28 19 Q 21 29 8 28 Q 0 23 2 8 Z",
          GOLD, width=3.0)
    _path(h, "M 29 6 Q 29 14 25 19", width=DETAIL)
    return h


def _kidney(g, x, y, mirror=False, scale=1):
    tr = f"translate({x} {y}) scale({-scale if mirror else scale} {scale})"
    h = _el(g, "g", transform=tr)
    _path(h, "M 15 1 C 3 -3 -2 11 1 21 C 3 31 14 34 19 27 "
          "Q 24 20 16 17 Q 11 15 16 11 C 21 8 20 3 15 1 Z",
          PURPLE, width=3.0)
    return h


def _bottle(g, x, y, width=34, height=65):
    b = _el(g, "g", transform=f"translate({x} {y}) scale({width/34:g} {height/65:g})")
    _path(b, "M 8 13 H 26 V 21 Q 32 25 32 32 V 59 "
          "Q 32 63 28 63 H 6 Q 2 63 2 59 V 32 Q 2 25 8 21 Z", PALE,
          width=3.2)
    _rect(b, 6, 2, 22, 13, BLUE, r=3, width=3.2)
    _rect(b, 4, 33, 26, 20, GREEN, r=1.5, width=0)
    _line(b, 11, 40, 23, 40, WHITE, 3)
    _line(b, 11, 46, 20, 46, WHITE, 3)
    return b


def _human():
    g = _group("human", "Human physiology template with heart, liver and kidneys",
               "translate(38 83) scale(.95 .95)")
    # Quiet, rounded silhouette: leg and arm contours remain legible at print size.
    _path(g, "M 69 45 V 55 L 48 62 Q 37 67 32 84 L 22 124 "
          "L 7 173 Q 4 185 12 188 Q 19 189 23 179 L 40 132 "
          "L 49 102 L 47 152 Q 43 170 45 184 L 44 219 H 70 "
          "L 77 182 H 85 L 92 219 H 118 L 117 184 Q 119 170 115 152 "
          "L 113 102 L 122 132 L 139 179 Q 143 189 150 188 "
          "Q 158 185 155 173 L 140 124 L 130 84 Q 125 67 114 62 "
          "L 93 55 V 45 Z", PALE, width=4)
    _path(g, "M 61 23 C 61 7 69 3 81 3 C 93 3 101 7 101 23 "
          "V 30 C 101 44 91 52 81 52 C 71 52 61 44 61 30 Z", WHITE,
          width=4)
    _line(g, 57, 68, 71, 64, BLUE, 3)
    _line(g, 91, 64, 105, 68, BLUE, 3)
    _heart(g, 73, 77, 1.2)
    _liver(g, 57, 115, 1.05)
    _kidney(g, 56, 155, scale=.9)
    _kidney(g, 106, 155, mirror=True, scale=.9)
    return g


def _evidence():
    g = _group("evidence", "Drug evidence: source documents, magnifier and medication vial")
    # Two paper silhouettes, with fewer, stronger marks than the former article texture.
    _path(g, "M 17 5 H 92 L 111 24 V 134 H 17 Z", PALE)
    _path(g, "M 92 5 V 24 H 111", BLUE, width=3)
    _rect(g, 31, 23, 43, 7, PURPLE, r=2, width=0)
    _line(g, 31, 41, 72, 41, INK, 3)
    _path(g, "M 45 24 H 116 L 140 48 V 147 Q 140 153 134 153 "
          "H 51 Q 45 153 45 147 Z", WHITE)
    _path(g, "M 116 24 V 48 H 140", BLUE, width=3)
    _rect(g, 59, 43, 40, 8, BLUE, r=2, width=0)
    for yy, end, color in [(61, 119, INK), (78, 123, GREEN), (95, 116, INK),
                            (113, 121, PURPLE), (131, 104, INK)]:
        _line(g, 60, yy, end, yy, color, 4 if color != INK else 3)
    _bottle(g, 2, 89, 40, 69)
    # Flat lens with restrained two-colour detail, not a reflective glass material.
    _path(g, "M 127 88 L 148 110 Q 153 115 148 120 Q 143 124 139 119 "
          "L 119 97 Z", CORAL, width=3.6)
    _circle(g, 110, 76, 26, BLUE, width=3.6)
    _circle(g, 110, 76, 19, WHITE, width=2.2)
    _line(g, 99, 71, 122, 71, GREEN, 3)
    _line(g, 99, 79, 117, 79, INK, 2.5)
    _line(g, 99, 86, 113, 86, INK, 2.5)
    return g


def _drug_scales():
    g = _group("drug_scales", "Medication and balance for the feasibility check")
    _bottle(g, 3, 25, 35, 71)
    _el(g, "ellipse", cx=44, cy=95, rx=10, ry=7, fill=CORAL,
        stroke=INK, stroke_width=3.0, transform="rotate(-20 44 95)")
    _line(g, 39, 96.5, 48, 93.5, INK, 2)
    _el(g, "ellipse", cx=58, cy=103, rx=10, ry=7, fill=BLUE,
        stroke=INK, stroke_width=3.0, transform="rotate(18 58 103)")
    _line(g, 54, 101.5, 62, 104.5, INK, 2)
    _rect(g, 94, 14, 10, 82, GOLD, r=3, width=3.2)
    _path(g, "M 84 97 Q 99 92 115 97 L 120 108 H 79 Z", GOLD, width=3.2)
    _line(g, 68, 17, 130, 17, INK, 6)
    _line(g, 68, 17, 130, 17, GOLD, 2.5)
    _circle(g, 99, 15, 6, GOLD, width=3.0)
    for x in [69, 129]:
        _path(g, f"M {x} 20 L {x-14} 61 M {x} 20 L {x+14} 61",
              width=DETAIL)
        _path(g, f"M {x-15} 61 H {x+15} Q {x+12} 75 {x} 75 "
              f"Q {x-12} 75 {x-15} 61 Z", GOLD, width=3.0)
    _path(g, "M 126 87 H 146 V 81 L 157 90 L 146 99 V 93 H 126 Z",
          GOLD, width=2.5)
    return g


def _lock(g, x, y, scale=1):
    h = _el(g, "g", transform=f"translate({x} {y}) scale({scale})")
    _path(h, "M 8 18 V 11 C 8 -1 26 -1 26 11 V 18", width=3.5)
    _rect(h, 2, 17, 30, 29, GOLD, r=4, width=3.5)
    _circle(h, 17, 29, 3, INK, width=0)
    _line(h, 17, 31, 17, 36, INK, 3.0)
    return h


def _blueprint():
    g = _group("blueprint", "Locked PBPK blueprint document and fixed model structure")
    _rect(g, 3, 15, 111, 103, BLUE, r=4)
    _path(g, "M 15 4 H 102 L 125 27 V 105 Q 125 111 119 111 H 21 "
          "Q 15 111 15 105 Z", WHITE)
    _path(g, "M 102 4 V 27 H 125", BLUE, width=3.2)
    _rect(g, 29, 18, 46, 6, BLUE, r=2, width=0)
    # A small compartment blueprint, with coloured components and visible links.
    _path(g, "M 65 42 V 57 H 37 V 66 M 65 57 H 91 V 66 "
          "M 37 80 V 91 H 65 M 91 80 V 91 H 65 V 96", width=2.3)
    _rect(g, 55, 31, 20, 16, CORAL, r=4, width=2.3)
    _rect(g, 26, 64, 22, 16, BLUE, r=4, width=2.3)
    _rect(g, 80, 64, 22, 16, GREEN, r=4, width=2.3)
    _rect(g, 55, 91, 20, 14, PURPLE, r=4, width=2.3)
    _lock(g, 108, 39, .92)
    return g


def _target_human():
    g = _group("target_human", "Target human and dosing through an infusion bag")
    _path(g, "M 19 23 V 29 L 11 33 Q 7 35 6 43 L 2 70 "
          "Q 1 76 6 76 L 12 51 V 65 L 9 87 H 22 L 26 67 "
          "H 28 L 33 87 H 45 L 42 65 V 51 L 48 76 Q 54 76 52 70 "
          "L 48 43 Q 47 35 42 33 L 34 29 V 23 Z", PALE, width=3.1)
    _path(g, "M 16 13 Q 16 2 27 2 Q 38 2 38 13 V 17 "
          "Q 37 28 27 28 Q 17 28 16 17 Z", BLUE, width=3.1)
    _heart(g, 21, 36, .49)
    _line(g, 21, 57, 33, 57, BLUE, 2.1)
    _line(g, 65, 11, 65, 87, INK, 3)
    _line(g, 58, 87, 72, 87, INK, 3)
    _path(g, "M 65 13 V 8 Q 65 3 71 3 Q 77 3 77 8 V 14", width=3)
    _rect(g, 65, 14, 17, 29, WHITE, r=4, width=3)
    _rect(g, 68, 28, 11, 12, GREEN, r=1.5, width=0)
    _path(g, "M 73 44 V 50 H 78 V 60 Q 78 70 68 70 H 53", width=2)
    return g


def _report():
    g = _group("report", "Concentration curve and endpoint report document")
    _rect(g, 3, 3, 76, 97, WHITE, r=5)
    _rect(g, 12, 12, 48, 6, BLUE, r=2, width=0)
    _path(g, "M 14 27 V 59 H 68", width=2)
    _path(g, "M 18 53 C 20 48 23 29 28 32 C 32 33 32 49 38 49 "
          "C 43 49 43 40 48 41 C 52 42 53 51 58 51 H 65",
          stroke=BLUE, width=2.8)
    for y, cols in [(68, [BLUE, GREEN, GOLD]), (81, [PALE, PALE, PALE])]:
        for x, color in zip([13, 32, 51], cols):
            _rect(g, x, y, 17, 10, color, r=1, width=1.3)
    return g


def _score():
    g = _group("score", "Endpoint scoring checklist and assessment result")
    _rect(g, 3, 11, 45, 66, WHITE, r=4, width=3.2)
    _path(g, "M 16 14 V 7 H 21 Q 21 2 26 2 Q 31 2 31 7 H 36 V 14 Z",
          PURPLE, width=3)
    for y, color, end in [(24, BLUE, 40), (37, GREEN, 36), (50, GOLD, 31)]:
        _rect(g, 10, y, 7, 7, color, r=1.5, width=1.7)
        _line(g, 23, y+3.5, end, y+3.5, INK, 2)
    _circle(g, 41, 60, 14, GREEN, width=3.2)
    _path(g, "M 33 60 L 39 65 L 49 54", width=3.3)
    return g


def flat_icon(name: str) -> ET.Element:
    """Return a standalone flat SVG group at its existing global position."""
    builders = {
        "human": _human, "evidence": _evidence, "drug_scales": _drug_scales,
        "blueprint": _blueprint, "target_human": _target_human,
        "report": _report, "score": _score,
    }
    try:
        return builders[name]()
    except KeyError:
        raise ValueError(f"Unknown flat blueprint icon: {name!r}") from None


def icon_sheet(output_dir: Path | None = None) -> Path:
    """Export a white-background SVG/PNG contact sheet, leaving workflow intact."""
    import cairosvg

    root = Path(__file__).resolve().parents[1]
    output_dir = output_dir or root / "results/figures/workflow/previews/flat_icons_20261026"
    output_dir.mkdir(parents=True, exist_ok=True)
    ET.register_namespace("", NS)
    canvas = _el(None, "svg", width=1120, height=650, viewBox="0 0 1120 650")
    _el(canvas, "rect", width=1120, height=650, fill=WHITE)
    _el(canvas, "title").text = "Flat scientific icon family for the PBPK blueprint workflow"
    for i, name in enumerate(ICON_BOUNDS):
        col, row = i % 4, i // 4
        x, y, w, h = ICON_BOUNDS[name]
        # Keep each original icon's aspect ratio; show all at a comparable size.
        scale = min(210 / w, 240 / h)
        cx, top = col*280 + 140, row*325 + 30
        wrap = _el(canvas, "g", transform=f"translate({cx-w*scale/2:g} {top+(240-h*scale)/2:g}) scale({scale:g}) translate({-x:g} {-y:g})")
        wrap.append(flat_icon(name))
        label = _el(canvas, "text", x=cx, y=top+276, fill=INK,
                    font_family="DejaVu Sans", font_size=18, text_anchor="middle")
        label.text = name.replace("_", " ")
    svg = output_dir / "icon_sheet.svg"
    ET.ElementTree(canvas).write(svg, encoding="utf-8", xml_declaration=True)
    cairosvg.svg2png(url=str(svg), write_to=str(output_dir / "icon_sheet.png"), scale=1.5)
    return svg


if __name__ == "__main__":
    print(icon_sheet())
