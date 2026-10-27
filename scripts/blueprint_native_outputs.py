"""Original editable SVG artwork for the four prediction-output illustrations.

The artwork is constructed from deliberately drawn Bézier paths, lines and
gradients.  No raster image, tracing output or image mask is used.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

NS = "http://www.w3.org/2000/svg"


def _element(tag: str, **attributes) -> ET.Element:
    return ET.Element(f"{{{NS}}}{tag}", {k.replace("_", "-"): str(v) for k, v in attributes.items()})


def _add(parent: ET.Element, tag: str, **attributes) -> ET.Element:
    node = _element(tag, **attributes)
    parent.append(node)
    return node


def _path(parent, d, fill="none", stroke="none", width=1, **kwargs):
    return _add(parent, "path", d=d, fill=fill, stroke=stroke, stroke_width=width,
                stroke_linejoin="round", stroke_linecap="round", **kwargs)


def _line(parent, x1, y1, x2, y2, stroke, width=1, **kwargs):
    return _add(parent, "line", x1=x1, y1=y1, x2=x2, y2=y2,
                stroke=stroke, stroke_width=width, stroke_linecap="round", **kwargs)


def _gradient(defs, name, stops, horizontal=False):
    grad = _add(defs, "linearGradient", id=name, x1="0%", y1="0%",
                x2="100%" if horizontal else "0%", y2="0%" if horizontal else "100%")
    for offset, color in stops:
        _add(grad, "stop", offset=offset, stop_color=color)


def _lock(parent, x, y, scale=1, prefix="bn_lock"):
    g = _add(parent, "g", transform=f"translate({x} {y}) scale({scale})")
    defs = _add(g, "defs")
    _gradient(defs, prefix+"_gold", [("0%", "#fffbe4"), ("50%", "#ffe9ac"), ("100%", "#f2bd64")])
    _path(g, "M 7 19 V 11 C 7 -2 25 -2 25 11 V 19", "none", "#8e470d", 3.1)
    _path(g, "M 8 19 V 11 C 8 0 24 0 24 11 V 19", "none", "#ffde9b", 1.5)
    _path(g, "M 12 18 V 11 C 12 4 20 4 20 11 V 18", "none", "#9b5215", 1.3)
    _add(g, "rect", x=1, y=17, width=31, height=29, rx=3.2, fill=f"url(#{prefix}_gold)", stroke="#8b480f", stroke_width=1.9)
    _path(g, "M 4 42 V 21 H 29", "none", "#fff9db", 1.2)
    _path(g, "M 19 28 C 19 24 13 24 13 28 C 13 30 14 31 15 31 L 14 37 H 19 L 18 31 C 19 30 19 29 19 28 Z", "#99551c")
    return g


def _blueprint():
    root = _element("g", id="native_output_blueprint", transform="translate(912 620)")
    defs = _add(root, "defs")
    _gradient(defs, "bn_sheet", [("0%", "#ffffff"), ("100%", "#effaff")])
    _gradient(defs, "bn_sheet_edge", [("0%", "#bdefff"), ("45%", "#4c87ef"), ("100%", "#1d46ad")], True)
    # Two offset, tilted drawing sheets: the blue edge remains visible.
    _path(root, "M 3 16 L 117 2 Q 120 2 121 6 L 132 103 Q 133 107 128 108 L 19 121 Q 15 122 14 117 Z", "#d5ebfa", "#334999", 1.8)
    _path(root, "M 4 19 L 12 18 L 23 115 L 17 117 Z", "url(#bn_sheet_edge)", "#2552bd", 1)
    _path(root, "M 13 17 L 116 3 Q 119 3 120 6 L 132 103 L 26 117 Z", "url(#bn_sheet)", "#2654a6", 1.8)
    _path(root, "M 24 25 L 107 14 L 119 95 L 36 106 Z", "#f8fcff", "#7ebbf3", 1.2)
    _path(root, "M 30 30 L 103 21 L 113 92 L 41 101 Z", "none", "#b7e4ff", .8)
    # Miniature model is drawn in its own tilted document coordinates.
    diagram = _add(root, "g", transform="matrix(1 -.12 .12 1 42 30)")
    _path(diagram, "M 17 1 H 39 V 69 H 17 Z M 5 22 H 50 V 45 H 5 Z M 17 46 H 4 V 57 H 40", "none", "#326fd2", 1.45)
    _path(diagram, "M 25 1 V 69 M 25 22 H 50 M 4 45 H 25", "none", "#6574cf", 1.1)
    # The heart, liver and kidneys give the document an anatomical center.
    _add(diagram, "ellipse", cx=25, cy=2, rx=6.3, ry=5.6, fill="#f6d6f3", stroke="#b967c7", stroke_width=.9)
    _path(diagram, "M 22 0 Q 24 -2 26 0 M 22 3 Q 25 1 28 3", "none", "#ce8ed3", .6)
    _path(diagram, "M 21 19 C 16 13 22 10 25 14 C 29 9 34 15 29 19 L 25 24 Z", "#f9b4d8", "#bd538f", .9)
    _path(diagram, "M 14 37 C 15 31 26 28 34 32 L 37 37 Q 31 42 21 40 Q 16 42 14 37 Z", "#f6ab69", "#c66c3a", .9)
    _path(diagram, "M 29 33 Q 26 37 27 40", "none", "#ce743f", .8)
    _path(diagram, "M 20 46 C 12 43 12 55 17 56 C 21 57 22 52 18 51 C 17 49 20 48 20 46 Z", "#eba5a6", "#ad5b6a", .8)
    _path(diagram, "M 29 46 C 37 43 37 55 32 56 C 28 57 27 52 31 51 C 32 49 29 48 29 46 Z", "#eba5a6", "#ad5b6a", .8)
    _path(diagram, "M 25 60 C 14 58 13 68 21 71 C 23 76 31 73 33 68 C 36 60 30 58 25 60 Z", "#eed8f3", "#9d6ab7", .8)
    for x,y,color in [(1,19,"#bde6f9"),(43,19,"#bde6f9"),(1,44,"#c4eccb"),(43,44,"#c4eccb"),(20,26,"#f0d0fa")]:
        _add(diagram, "rect", x=x, y=y, width=9, height=10, rx=2, fill=color, stroke="#5387b9", stroke_width=.8)
    _lock(root, 106, 22, .9, "bn_blueprint_lock")
    return root


def _human():
    root = _element("g", id="native_output_target_human", transform="translate(1115 604)")
    defs = _add(root, "defs")
    _gradient(defs, "bn_body", [("0%", "#effeff"), ("25%", "#b2eafb"), ("57%", "#78c5f2"), ("82%", "#d3f5ff"), ("100%", "#6cafe7")], True)
    _gradient(defs, "bn_iv", [("0%", "#e6fcff"), ("48%", "#96d9ff"), ("100%", "#336ec9")], True)
    _gradient(defs, "bn_face", [("0%", "#edffff"), ("100%", "#93d2f5")], True)
    body = "M 28 3 C 20 3 19 9 21 16 C 18 18 21 22 23 23 L 22 28 C 19 31 13 30 11 36 L 6 55 L 4 73 L 2 85 Q 3 88 6 88 L 9 76 L 11 58 L 15 47 L 15 68 Q 15 77 13 88 L 24 88 L 27 70 L 29 70 L 32 88 L 43 88 Q 41 76 41 68 L 41 47 L 45 58 L 47 76 L 50 88 Q 53 88 54 85 L 52 72 L 50 54 L 45 36 Q 44 32 34 29 L 33 23 Q 37 19 35 16 C 37 8 35 3 28 3 Z"
    _path(root, body, "url(#bn_body)", "#164ca4", 1.6)
    _path(root, "M 27 6 C 22 6 22 12 24 18 L 25 22 L 30 22 Q 34 17 33 12 C 33 7 30 6 27 6 Z", "url(#bn_face)")
    _path(root, "M 14 36 Q 23 30 28 34 Q 35 30 43 36 M 28 35 V 56 M 21 58 Q 28 61 35 58 M 19 66 Q 28 70 38 66", "none", "#448ec7", 1.15, opacity=".6")
    _path(root, "M 19 38 Q 15 51 18 61 L 17 83 M 38 38 Q 41 50 38 61 L 39 83 M 26 70 L 24 86 M 30 70 L 32 86", "none", "#e4fbff", 1.3, opacity=".9")
    _path(root, "M 10 41 L 8 57 L 6 75 M 47 42 L 49 58 L 51 75", "none", "#ecffff", 1)
    # IV stand, hooked bag, drip chamber and softly curved delivery tubing.
    _line(root, 65, 13, 65, 85, "#163c91", 3)
    _line(root, 65, 13, 65, 85, "#8bd0f4", 1)
    _path(root, "M 65 13 V 8 Q 65 3 70 3 Q 74 3 74 9 V 14", "none", "#204c9d", 1.7)
    _path(root, "M 67 11 L 79 11 L 84 17 V 39 Q 84 44 80 44 H 68 Q 63 44 63 39 V 17 Q 63 13 67 11 Z", "url(#bn_iv)", "#134b9c", 1.6)
    _path(root, "M 67 16 H 80 V 38 H 67 Z", "#fffefe", "#7ba8e7", .7)
    _path(root, "M 71 24 Q 74 19 78 25 Q 75 28 71 26 Z", "#ff397c")
    _path(root, "M 70 39 H 79", "none", "#85dbfa", 2)
    _path(root, "M 71 44 V 48 H 77 V 44 M 74 48 V 65 Q 74 79 68 85", "none", "#174ba2", 1.6)
    _path(root, "M 75 50 V 66 Q 75 79 68 84", "none", "#b2f0ff", .75)
    _path(root, "M 68 18 V 34", "none", "white", 1.2)
    return root


def _report():
    root = _element("g", id="native_output_report", transform="translate(1484 630)")
    defs = _add(root, "defs")
    _gradient(defs, "bn_report_paper", [("0%", "#ffffff"), ("100%", "#f1f9ff")])
    _gradient(defs, "bn_report_chart", [("0%", "#dcefff"), ("100%", "#f9fdff")])
    _add(root, "rect", x=3, y=3, width=76, height=97, rx=5, fill="#d8e6ef", stroke="#152252", stroke_width=2.2)
    _add(root, "rect", x=5, y=5, width=72, height=93, rx=3.5, fill="url(#bn_report_paper)", stroke="#6d72c4", stroke_width=.9)
    _add(root, "rect", x=16, y=17, width=51, height=40, fill="url(#bn_report_chart)")
    for y in [23,31,39,47,55]:
        _line(root,16,y,67,y,"#d6e6f9",.6)
    for x in [25,35,45,55,65]:
        _line(root,x,17,x,57,"#e3eef8",.6)
    _path(root, "M 15 16 V 57 H 68", "none", "#597fbb", 1)
    _path(root, "M 19 48 C 21 45 21 32 25 29 C 29 23 31 41 35 43 C 39 47 41 37 44 34 C 47 30 49 40 52 43 C 55 46 56 39 59 40 C 61 40 61 44 64 44", "none", "#155bea", 1.7)
    for y in [65,75,85]:
        for x,w in [(15,16),(33,17),(52,16)]:
            _add(root, "rect", x=x,y=y,width=w,height=8,fill="#8baecc" if y!=75 else "#a1bfd6", stroke="#709bbd",stroke_width=.25)
    _line(root, 10,9,72,9,"white",1.2)
    return root


def _score():
    root = _element("g", id="native_output_score", transform="translate(1507 826)")
    defs = _add(root, "defs")
    _gradient(defs, "bn_score_paper", [("0%", "#ffffff"), ("100%", "#f2f0ff")])
    _gradient(defs, "bn_score_green", [("0%", "#01d58a"), ("100%", "#00a552")])
    _add(root, "rect", x=4,y=9,width=46,height=68,rx=3,fill="url(#bn_score_paper)",stroke="#301483",stroke_width=2.6)
    _path(root, "M 17 13 V 6 H 22 C 22 -1 30 -1 31 6 H 36 V 13 Z", "#eee9ff", "#291278", 1.8)
    _add(root, "circle", cx=26.5,cy=6,r=1.9,fill="white",stroke="#30208c",stroke_width=1)
    _path(root, "M 11 23 L 14 20 L 17 23 L 14 26 Z M 11 32 L 13 30 L 15 32 L 13 34 Z", "#5139a6")
    _line(root, 21,23,41,23,"#8b80b5",1.4)
    _line(root, 20,32,38,32,"#c0b5d6",1.2)
    _line(root, 11,41,38,41,"#d2cbe2",1)
    _line(root, 11,48,34,48,"#d2cbe2",1)
    _add(root,"circle",cx=40,cy=51,r=17.1,fill="#0b3879",stroke="#14266a",stroke_width=1.2)
    _add(root,"circle",cx=40,cy=51,r=14.8,fill="url(#bn_score_green)",stroke="#96f0d2",stroke_width=.9)
    _path(root,"M 32 50 L 38 56 L 49 43", "none", "white",3.6)
    _path(root,"M 30 43 Q 36 35 45 39", "none", "#77edba",.85)
    return root


def native_output(name: str) -> ET.Element:
    """Return one SVG group positioned in the figure's 1605 × 980 canvas."""
    builders = {"blueprint": _blueprint, "target_human": _human,
                "report": _report, "score": _score}
    try:
        return builders[name]()
    except KeyError:
        raise ValueError(f"Unknown native output illustration: {name!r}") from None

