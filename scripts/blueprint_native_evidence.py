"""Hand-drawn native SVG evidence artwork for the PBPK workflow.

Geometry, smooth curves, gradients and highlights are authored explicitly.
No raster input, contour tracing, pixel paths, image embedding or cache import.
``native_evidence`` returns a self-contained SVG group in 1605×980 coordinates.
"""
from __future__ import annotations

import itertools
import xml.etree.ElementTree as ET

NS = "http://www.w3.org/2000/svg"
_SERIAL = itertools.count()


def _el(parent, tag, **attrs):
    return ET.SubElement(parent, f"{{{NS}}}{tag}", {
        key.replace("_", "-"): str(value) for key, value in attrs.items()
    })


def _path(parent, d, fill="none", stroke="none", width=1, **attrs):
    return _el(parent, "path", d=d, fill=fill, stroke=stroke,
               stroke_width=width, **attrs)


def _line(parent, x1, y1, x2, y2, color, width=1, **attrs):
    return _el(parent, "line", x1=x1, y1=y1, x2=x2, y2=y2,
               stroke=color, stroke_width=width, **attrs)


def _palette(defs, prefix):
    colors = {}
    for name, stops, vertical in [
        ("paper", [(0,"#ffffff"),(75,"#fafbfc"),(100,"#e1e8ed")], True),
        ("cap", [(0,"#34566c"),(15,"#9bb8c9"),(36,"#f2f7fc"),(64,"#a7becd"),(88,"#50758e"),(100,"#24475f")], False),
        ("glass", [(0,"#07668f"),(9,"#45c0f3"),(23,"#d9fbff"),(38,"#80d7f9"),(64,"#4bc2ef"),(87,"#139cda"),(100,"#165782")], False),
        ("liquid", [(0,"#91e2ff"),(25,"#a4e5ff"),(78,"#5ecdf8"),(100,"#0e90cf")], False),
        ("gold", [(0,"#9b5715"),(18,"#e9ae38"),(39,"#fff0ad"),(60,"#efbe4d"),(86,"#c68c22"),(100,"#8e511c")], False),
        ("gold_h", [(0,"#fff2b3"),(42,"#edba44"),(100,"#ba7a23")], True),
        ("handle", [(0,"#ffe29c"),(30,"#dfab51"),(67,"#b87828"),(100,"#643912")], False),
        ("redpill", [(0,"#ffe5ec"),(32,"#ff9eb6"),(100,"#d84b75")], True),
        ("whitepill", [(0,"#ffffff"),(52,"#f6fcff"),(100,"#bedbf0")], True),
    ]:
        gid=prefix+name
        grad=_el(defs,"linearGradient",id=gid,x1="0%",y1="0%",
                 x2="0%" if vertical else "100%",y2="100%" if vertical else "0%")
        for offset,color in stops:
            _el(grad,"stop",offset=f"{offset}%",stop_color=color)
        colors[name]=f"url(#{gid})"
    for name, mid, dark in [
        ("carbon","#91a4b9","#25384b"), ("hydrogen","#e4dbd2","#746864"),
        ("nitrogen","#2fb8f2","#00649b"), ("oxygen","#ff676d","#b00018"),
    ]:
        gid=prefix+name
        grad=_el(defs,"radialGradient",id=gid,cx="31%",cy="24%",r="77%")
        for offset,color in [(0,"#fffef5"),(24,mid),(100,dark)]:
            _el(grad,"stop",offset=f"{offset}%",stop_color=color)
        colors[name]=f"url(#{gid})"
    return colors


def _bottle(parent, x, y, w, h, c):
    """Rounded glass bottle in a 44×92 local drawing coordinate system."""
    g=_el(parent,"g",transform=f"translate({x},{y}) scale({w/44},{h/92})")
    _path(g,"M10 19 L10 25 Q10 28 7 31 Q2 36 2 43 L2 82 "
          "Q2 89 22 89 Q42 89 42 82 L42 43 Q42 36 37 31 "
          "Q34 28 34 25 L34 19 Z",c["glass"],"#174867",1.6)
    _path(g,"M6 47 L38 47 L38 82 Q38 85.5 22 86 Q6 85.5 6 82 Z",c["liquid"],"none",opacity=.8)
    _el(g,"ellipse",cx=22,cy=47,rx=16,ry=3.3,fill="#8de4ff",stroke="#55b5e0",stroke_width=.7)
    _path(g,"M8.4 39 Q9 34 13 31 L13 27",stroke="#eaffff",width=3.6,opacity=.92)
    _path(g,"M9 48 V81 Q9 83 15 84",stroke="#e9fdff",width=4.1,opacity=.88)
    _path(g,"M13 49 V80",stroke="#fff",width=.85,opacity=.85)
    _path(g,"M35 40 V82",stroke="#1a8ec7",width=1.2,opacity=.8)
    _path(g,"M4.5 81 Q22 75.3 39.5 81",stroke="#45a9d7",width=1.7,opacity=.8)
    _el(g,"ellipse",cx=22,cy=82.5,rx=19.2,ry=5.1,fill="none",stroke="#beeaff",stroke_width=1.1)
    _path(g,"M6 85 Q22 90 38 85",stroke="#104f76",width=.85)
    _path(g,"M7 6.4 V18 Q7 22.5 22 22.5 Q37 22.5 37 18 V6.4Z",c["cap"],"#183e59",1.4)
    _el(g,"ellipse",cx=22,cy=6.4,rx=15,ry=4.3,fill="#8ad4f5",stroke="#214a68",stroke_width=1.3)
    _el(g,"ellipse",cx=22,cy=5.9,rx=12.2,ry=2.5,fill="#b1e8ff",stroke="none")
    _path(g,"M9.3 17.8 Q22 21.3 34.7 17.8",stroke="#e4f5ff",width=1.1)
    return g


def _evidence(g, c):
    # Two individually drawn research papers, aligned to the selected layout.
    rear=_el(g,"g",transform="translate(7,18) rotate(-12.3)")
    _el(rear,"rect",x=0,y=0,width=78,height=124.5,rx=.8,
        fill=c["paper"],stroke="#5e6b78",stroke_width=1.05)
    _line(rear,7,8,70,8,"#b4bbc0",1.7)
    _line(rear,7,12,59,12,"#c1c6ca",1.2)
    # Greyscale anatomical figure on the back page.
    _el(rear,"ellipse",cx=22,cy=25,rx=4.7,ry=7.3,fill="#d0d4d6",stroke="#bec4c8",stroke_width=.7)
    _path(rear,"M18.5 32 L18 35 Q12 37 9.5 41 L7.5 60 L10.5 67 L13 64 "
          "L15 45 L16.5 64 L15.5 83 L20 85 L22 68 L24 85 L28 84 L27 64 "
          "L28.5 45 L31.5 64 L34 65 L35 61 L31.5 42 Q30 37 25.5 35 L25 32 Z",
          "#c8cdd0","#b8c0c5",.7)
    _path(rear,"M22 38 V66 M16 42 Q22 38 28 42 M18 48 Q22 45 26 48 "
          "M18 53 Q22 49 26 53 M18 57 Q22 53 26 57",stroke="#e8ecee",width=.8)
    for i in range(17):
        yy=20+4.4*i
        _line(rear,40,yy,70-(i%4)*2,yy,"#b6bdc2",.85)
    for i in range(6):
        _line(rear,8,96+4*i,69-(i%3)*5,96+4*i,"#bec5c9",.9)
    # Folded top document clip / tab visible above the sheet.
    _path(g,"M17.1 9.6 Q16.8 7.2 19.4 6.7 L27.3 5.1 Q29.3 4.8 29.8 7.2 "
          "L31.4 14.5 L18.6 17.2 Z","#eef1f3","#8794a0",.9)
    _path(g,"M19 8.1 L26.7 6.5",stroke="#fff",width=1.0)
    front=_el(g,"g",transform="translate(64.5,14) rotate(9.5)")
    _el(front,"rect",x=0,y=0,width=88,height=122.4,rx=.9,
        fill=c["paper"],stroke="#4f5d6c",stroke_width=1.1)
    for yy,end,weight in [(8,60,1.8),(12,65,1.2),(17,79,1.0),(21,77,.9)]:
        _line(front,7,yy,end,yy,"#aeb6bc",weight)
    # Article thumbnail with a small grey organ/profile and graph axes.
    _el(front,"rect",x=59,y=27,width=22,height=23,rx=.3,
        fill="#e1e5e7",stroke="#c8ced2",stroke_width=.45)
    _path(front,"M61 47 C61 34 65 31 70 31 C77 31 79 37 78 47 Z",
          "#b4bcc3","#c2c9ce",.45)
    _line(front,68,36,68,46,"#d8dfe3",.7)
    for i in range(6):
        _line(front,8,28+i*4.2,54-(i%3)*3,28+i*4.2,"#b3bcc2",1)
    for i in range(16):
        _line(front,8,56+i*3.7,78-(i%4)*2.2,56+i*3.7,"#b0b9c0",.95)
    # Gold-rimmed optical magnifier, preserving visible paper lines underneath.
    _path(g,"M102 61.9 L128.7 86.2",stroke="#45301f",width=7.4)
    _path(g,"M102 61.5 L128.2 85.7",stroke=c["handle"],width=5.0)
    _path(g,"M104.1 62.3 L126.8 83.8",stroke="#f9d590",width=1.1)
    _path(g,"M126 84.3 L128.5 86.2",stroke="#70451c",width=2.8)
    _el(g,"ellipse",cx=87.7,cy=44.9,rx=18.3,ry=21.2,
        fill="#eef7ff",fill_opacity=.24,stroke="#23313b",stroke_width=5.6,
        transform="rotate(-5,87.7,44.9)")
    _el(g,"ellipse",cx=87.7,cy=44.9,rx=18.3,ry=21.2,
        fill="none",stroke=c["gold"],stroke_width=3.05,
        transform="rotate(-5,87.7,44.9)")
    _el(g,"ellipse",cx=87.7,cy=44.9,rx=15.8,ry=18.7,
        fill="none",stroke="#7b8b98",stroke_width=.7,
        transform="rotate(-5,87.7,44.9)")
    _path(g,"M77 32 Q84 23 95 28",stroke="#fff",width=1.7,opacity=.8)
    _path(g,"M101 47 Q100 57 91 62",stroke="#d9e8f3",width=1.4,opacity=.78)
    _bottle(g,2.0,62.2,40.4,81.9,c)
    # Explicit ball-and-stick geometry.  Double highlight strokes give the
    # narrow bonds a metallic cylindrical appearance at publication scale.
    atoms=[
        (50,124.7,5.0,"carbon"),(60.1,110.5,5.0,"carbon"),
        (75.2,117.0,5.4,"nitrogen"),(76.1,128.3,5.2,"carbon"),
        (61.4,128.4,5.2,"carbon"),(55.0,138.0,4.8,"carbon"),
        (87.8,138.4,5.2,"carbon"),(92.3,153.3,5.3,"hydrogen"),
        (49.4,150.2,5.2,"hydrogen"),(84.4,103.3,5.1,"carbon"),
        (81.2,89.6,4.6,"hydrogen"),(101.1,106.2,5.2,"oxygen"),
        (101.8,128.6,5.4,"oxygen"),(56.2,97.8,4.5,"hydrogen"),
        (42.4,117.4,4.5,"hydrogen"),
    ]
    bonds=[(0,1),(1,2),(2,3),(3,4),(4,0),(4,5),(5,8),(3,6),(6,7),
           (6,12),(2,9),(9,10),(9,11),(1,13),(0,14)]
    for ia,ib in bonds:
        ax,ay,*_=atoms[ia];bx,by,*_=atoms[ib]
        _line(g,ax,ay,bx,by,"#172938",4)
        _line(g,ax-.35,ay-.45,bx-.35,by-.45,"#c2d0dc",1.55)
        _line(g,ax-.7,ay-.75,bx-.7,by-.75,"#f8faff",.55)
    for ax,ay,r,kind in atoms:
        _el(g,"circle",cx=ax,cy=ay,r=r,fill=c[kind],stroke="#20303c",stroke_width=.9)
        _el(g,"ellipse",cx=ax-r*.23,cy=ay-r*.3,rx=r*.29,ry=r*.16,
            fill="#fff",opacity=.65,stroke="none")


def _drug_scales(g, c):
    # Reference bottle and three overlapping medication tablets.
    _bottle(g,7.0,14.0,36.8,80.2,c)
    _el(g,"ellipse",cx=50,cy=71.7,rx=9.4,ry=7.3,fill=c["redpill"],
        stroke="#95385c",stroke_width=1.1,transform="rotate(-16,50,71.7)")
    _path(g,"M44.2 68.7 Q49.4 64.5 54 68",stroke="#ffe7f0",width=1.15)
    _el(g,"ellipse",cx=61.5,cy=85,rx=10.0,ry=7.4,fill=c["redpill"],
        stroke="#9c3155",stroke_width=1.1,transform="rotate(9,61.5,85)")
    _path(g,"M56 82 Q60 79 65 81.3",stroke="#ffcadc",width=1)
    _el(g,"ellipse",cx=39.7,cy=88,rx=9.8,ry=8.1,fill=c["whitepill"],
        stroke="#1e5a8f",stroke_width=1.2,transform="rotate(-15,39.7,88)")
    _path(g,"M34.3 85 Q38.6 80.9 44 83",stroke="#fff",width=1.4)
    # Gold pedestal: layered foot and rounded metallic rim.
    _el(g,"ellipse",cx=105.5,cy=101.7,rx=22.9,ry=8.8,
        fill="#69421d",stroke="#4b3c25",stroke_width=1)
    _path(g,"M83 98.5 V102 Q84 109.4 105.5 109.4 Q128.2 109.4 128.3 102 V98.5 Z",
          c["gold"],"#68461d",1)
    _el(g,"ellipse",cx=105.5,cy=98.6,rx=22.8,ry=7.1,
        fill=c["gold_h"],stroke="#665027",stroke_width=1.2)
    _el(g,"ellipse",cx=105.5,cy=98.3,rx=17.3,ry=4.6,
        fill="#d89b2e",stroke="#f7d16e",stroke_width=.75)
    # Central turned column, flared foot and spherical pivot.
    _path(g,"M100.5 90 L102.2 16 L108 16 L110 90 Q116.3 93.2 112.6 96 "
          "Q105.7 98.4 98.7 96 Q95 93.3 100.5 90 Z",
          c["gold"],"#755222",1)
    _path(g,"M104.3 19 L103.5 89 Q102.8 92.3 100.4 93",stroke="#fff1bc",width=1.45)
    _path(g,"M107.5 20 L108.2 89",stroke="#b47416",width=.7)
    # Horizontal balance beam with rounded gold ends and fine inner highlight.
    _path(g,"M83.7 10.1 L126.3 10.1 Q128.1 10.1 128.1 12.3 "
          "Q128 14.6 126.2 14.6 L83.8 14.6 Q81.8 14.6 81.8 12.3 Q81.8 10.1 83.7 10.1Z",
          c["gold_h"],"#6e4a21",1)
    _line(g,84.1,11.3,125.4,11.3,"#fff3b8",1)
    _el(g,"ellipse",cx=105.6,cy=10.6,rx=3.6,ry=4.8,
        fill=c["gold_h"],stroke="#805224",stroke_width=1)
    _el(g,"ellipse",cx=104.6,cy=8.9,rx=1.25,ry=1.6,fill="#fff3bd")
    # Suspension chains: thin dark outlines with luminous metal cores.
    for cx in (83.8,126.2):
        for endx in (cx-13,cx+13):
            _line(g,cx,15,endx,57.1,"#514c3d",1.25)
            _line(g,cx+.25,15,endx+.25,57.1,"#d7ba74",.45)
        _line(g,cx,16,cx,56.8,"#856328",.75)
        # Shallow concave pans, heavy rim and bright interior edge.
        _path(g,f"M{cx-14} 57 Q{cx} 51.3 {cx+14} 57 Q{cx+12} 65.9 {cx} 66.1 "
              f"Q{cx-12} 65.9 {cx-14} 57Z",c["gold_h"],"#75501e",1.05)
        _el(g,"ellipse",cx=cx,cy=57.2,rx=13.6,ry=3.3,
            fill="#ffe799",stroke="#8f621c",stroke_width=.8)
        _path(g,f"M{cx-10.5} 56.3 Q{cx} 53.4 {cx+10.7} 56.4",stroke="#fff5c3",width=1)
        _path(g,f"M{cx-8.5} 63 Q{cx} 66 {cx+8.5} 63",stroke="#ba7f1b",width=.7)
    # The original small arrow leads from medication/scale to feasibility.
    _path(g,"M124 79.3 H146.3 V74.7 L158.2 82 L146.3 89.1 V84.7 H124Z",
          c["gold_h"],"#ad791e",.65)


def native_evidence(name: str) -> ET.Element:
    """Return evidence or drug_scales, already positioned in global coordinates.

    evidence: (220,122)–(374,283); drug_scales: (35,657)–(194,772).
    Each call gets unique gradient IDs, allowing repeated independent instances.
    """
    if name not in {"evidence","drug_scales"}:
        raise ValueError(f"Unsupported native evidence illustration: {name!r}")
    x,y=(220,122) if name=="evidence" else (35,657)
    prefix=f"native-evidence-{name}-{next(_SERIAL)}-"
    g=ET.Element(f"{{{NS}}}g",{
        "id":prefix+"art", "transform":f"translate({x},{y})",
        "stroke-linecap":"round", "stroke-linejoin":"round",
        "aria-label":"Hand-drawn "+name.replace("_"," "),
    })
    defs=_el(g,"defs")
    colors=_palette(defs,prefix)
    if name=="evidence":
        _evidence(g,colors)
    else:
        _drug_scales(g,colors)
    return g
