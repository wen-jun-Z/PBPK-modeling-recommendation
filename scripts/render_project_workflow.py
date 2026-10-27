"""Render the project's current workflow using only its local font and CPU.

SVG is the editable vector figure; PNG is the README/browser preview.
Keep application branches distinct from the evidence-derived research inputs.
"""
from pathlib import Path
from math import atan2, cos, sin
import json
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/figures/workflow'
FONT = ROOT / 'runtime/fonts/NotoSansCJKsc-Regular.otf'
W, H, SCALE = 1800, 1540, 2
INK, MUTED, LINE = '#20354A', '#64758A', '#9DADBD'
TEAL, BLUE, PURPLE, AMBER = '#26796D', '#426CA0', '#71629B', '#AA752C'
BACKGROUND = '#F7F9FC'
ET.register_namespace('', 'http://www.w3.org/2000/svg')
SVG = '{http://www.w3.org/2000/svg}'
root = ET.Element(SVG+'svg', {'viewBox':f'0 0 {W} {H}', 'width':str(W), 'height':str(H),
    'role':'img', 'aria-labelledby':'title description'})
ET.SubElement(root, SVG+'title', {'id':'title'}).text = 'PBPK knowledge base and model recommendation: current workflow'
ET.SubElement(root, SVG+'desc', {'id':'description'}).text = (
    'Evidence and models are collected separately; user requests enter component or public-model recommendation.'
    'Current external experiments use evidence-derived templates with model descriptions or graphs; independent scientific evaluation remains incomplete.')
canvas = Image.new('RGB', (W*SCALE,H*SCALE), BACKGROUND)
draw = ImageDraw.Draw(canvas)
fonts, boxes = {}, []


def font(size):
    if size not in fonts:
        fonts[size] = ImageFont.truetype(str(FONT), int(size*SCALE))
    return fonts[size]


def text(x, y, value, size=25, color=INK, limit=None):
    # Keep translated labels on their original rows without clipping the cards.
    allowed_width = min(limit if limit is not None else W-x, W-x)
    while size > 14 and draw.textlength(value, font=font(size))/SCALE > allowed_width:
        size -= 1
    bounds = draw.textbbox((0,0),value,font=font(size),anchor='lt')
    width = (bounds[2]-bounds[0])/SCALE
    height = (bounds[3]-bounds[1])/SCALE
    if limit is not None and width > limit:
        raise ValueError(f'Text exceeds card width: {value}')
    if x < 0 or y < 0 or x+width > W or y+height > H:
        raise ValueError(f'Text outside figure: {value}')
    draw.text((x*SCALE,y*SCALE),value,font=font(size),fill=color,anchor='lt')
    ET.SubElement(root,SVG+'text',{'x':str(x),'y':str(y),'font-size':str(size),
        'font-family':'Noto Sans CJK SC, Noto Sans SC, sans-serif',
        'dominant-baseline':'text-before-edge','fill':color}).text=value
    boxes.append((x,y,width,height))


def polyline(points, color=LINE, width=2.2, dashed=False, arrow=False):
    attributes={'points':' '.join(f'{x},{y}' for x,y in points), 'fill':'none',
                'stroke':color,'stroke-width':str(width),'stroke-linejoin':'round'}
    if dashed: attributes['stroke-dasharray']='7 6'
    ET.SubElement(root,SVG+'polyline',attributes)
    for (x1,y1),(x2,y2) in zip(points,points[1:]):
        distance=((x2-x1)**2+(y2-y1)**2)**.5
        if not distance: continue
        if dashed:
            for start in range(0,int(distance),13):
                end=min(start+7,distance)
                draw.line(((x1+(x2-x1)*start/distance)*SCALE,(y1+(y2-y1)*start/distance)*SCALE,
                           (x1+(x2-x1)*end/distance)*SCALE,(y1+(y2-y1)*end/distance)*SCALE),
                          fill=color,width=round(width*SCALE))
        else: draw.line((x1*SCALE,y1*SCALE,x2*SCALE,y2*SCALE),fill=color,width=round(width*SCALE))
    if arrow:
        x,y=points[-1];px,py=points[-2];a=atan2(y-py,x-px)
        triangle=[(x,y),(x-11*cos(a)+5*sin(a),y-11*sin(a)-5*cos(a)),
                       (x-11*cos(a)-5*sin(a),y-11*sin(a)+5*cos(a))]
        ET.SubElement(root,SVG+'polygon',{'points':' '.join(f'{a},{b}' for a,b in triangle),'fill':color})
        draw.polygon([(a*SCALE,b*SCALE) for a,b in triangle],fill=color)


def rect(x,y,w,h,fill='#FFFFFF',stroke='#DCE4ED',radius=16,dashed=False):
    attributes={'x':str(x),'y':str(y),'width':str(w),'height':str(h),'rx':str(radius),
                'fill':fill,'stroke':stroke,'stroke-width':'1.5'}
    if dashed: attributes['stroke-dasharray']='8 6'
    ET.SubElement(root,SVG+'rect',attributes)
    draw.rounded_rectangle((x*SCALE,y*SCALE,(x+w)*SCALE,(y+h)*SCALE),radius=radius*SCALE,
                           fill=fill,outline=None if dashed else stroke,width=3)
    if dashed:
        # Dashed perimeter in the bitmap; the vector retains rounded corners.
        for points in ([(x+radius,y),(x+w-radius,y)],[(x+w,y+radius),(x+w,y+h-radius)],
                       [(x+radius,y+h),(x+w-radius,y+h)],[(x,y+radius),(x,y+h-radius)]):
            for (x1,y1),(x2,y2) in zip(points,points[1:]):
                length=abs(x2-x1)+abs(y2-y1)
                for start in range(0,int(length),14):
                    end=min(start+8,length)
                    draw.line(((x1+(x2-x1)*start/length)*SCALE,(y1+(y2-y1)*start/length)*SCALE,
                               (x1+(x2-x1)*end/length)*SCALE,(y1+(y2-y1)*end/length)*SCALE),
                              fill=stroke,width=3)


def card(x,y,w,h,title,lines,color=TEAL,pending=False,size=25):
    rect(x,y,w,h,fill='#FEF9F0' if pending else '#FFFFFF',
         stroke='#D9B981' if pending else '#DCE4ED',dashed=pending)
    rect(x+18,y+24,4,h-48,fill=color,stroke=color,radius=2)
    text(x+38,y+23,title,size=30,color=color,limit=w-66)
    for i,line in enumerate(lines):
        line_y=y+69+34*i
        if line_y+size > y+h-12: raise ValueError(f'Text exceeds card height: {title}')
        text(x+38,line_y,line,size=size,color=INK,limit=w-64)


def section(number,label,y,color):
    rect(60,y,42,38,fill=color,stroke=color,radius=10)
    text(71,y+7,str(number),size=24,color='#FFFFFF')
    text(118,y+2,label,size=31,color=INK)


rect(0,0,W,H,fill=BACKGROUND,stroke=BACKGROUND,radius=0)
text(60,48,'PBPK knowledge base and model recommendation',size=49)
text(62,118,'Current workflow / Data, existing functions, and research validation',size=26,color=MUTED)
text(1535,125,'2026 · 09 · 16',size=22,color=MUTED)

section(1,'Data collection and knowledge base',181,TEAL)
card(60,240,300,145,'Public evidence',['Literature · drug labels','Clinical PK data'],size=24)
card(430,240,530,145,'Collect, extract, and validate',['Rules + Qwen assistance','Attribution · deduplication · conflicts · provenance'],size=24)
card(1030,240,710,145,'Evidence store and search index',['30 drugs · 2,872 usable evidence records','Preserve source locations and citations'],size=25)
card(60,410,300,130,'Public PBPK models',['Model code and documentation'],size=24)
card(430,410,530,130,'Acquire, parse, and register models',['Versions · checksums · licenses · structure'],size=24)
card(1030,410,710,130,'Model catalog and structure graphs',['202 model families · 280 versions'],size=25)
for y in (312,475):
    polyline([(366,y),(422,y)],arrow=True)
    polyline([(966,y),(1022,y)],arrow=True)

section(2,'Existing recommendation functions: two branches',591,BLUE)
card(60,750,300,150,'User modeling request',['Drug · population · route','Prediction task and constraints'],color=BLUE,size=23)
card(510,670,560,145,'Component recommendation',['Retrieve and rerank → Qwen generation','Citation and rule validation'],color=BLUE,size=25)
card(1160,670,580,145,'Evidence-supported components',['Candidate mechanisms, components, and parameters','Sources · conflicts · uncertainty'],color=BLUE,size=25)
card(510,880,560,145,'Public-model recommendation',['Request profile + model catalog + mechanism evidence','Relevance and intended-use gates → tiered ranking'],color=BLUE,size=24)
card(1160,880,580,175,'Candidates and applicability boundaries',['Reuse / adapt / validate / reference / reject','Same-drug executable templates can enter','Blueprint generation and execution'],color=BLUE,size=24)
polyline([(366,825),(412,825)])
polyline([(412,742),(412,952)])
polyline([(412,742),(502,742)],arrow=True)
polyline([(412,952),(502,952)],arrow=True)
polyline([(1078,742),(1152,742)],arrow=True)
polyline([(1078,952),(1152,952)],arrow=True)
polyline([(760,822),(760,871)],color=BLUE,dashed=True,arrow=True)
text(785,838,'Component evidence (optional interface)',size=21,color=MUTED)

section(3,'Current model-recommendation experiments and next steps',1110,PURPLE)
card(60,1185,370,220,'Frozen experimental inputs',['772 requests × 280 versions','Evidence-derived request templates','Candidate text / metadata graphs'],color=PURPLE,size=24)
card(480,1185,370,220,'External-method comparisons',['Embeddings / rerankers / graph methods','12 configurations · zero-shot','Original / reordered / missing evidence'],color=PURPLE,size=24)
card(900,1185,370,220,'Currently measurable outcomes',['Ranking changes and output validity','Latency / GPU memory / failure causes','Retain original and corrected results'],color=PURPLE,size=24)
card(1320,1185,420,220,'Independent scientific evaluation · pending',['Add independent labels and study matches','Then evaluate ranking, PK error, rejection,','and confidence calibration'],color=AMBER,pending=True,size=24)
polyline([(438,1290),(472,1290)],arrow=True)
polyline([(858,1290),(892,1290)],arrow=True)
polyline([(1278,1290),(1312,1290)],color=AMBER,dashed=True,arrow=True)
text(62,1444,'Current comparisons use evidence-derived templates without a unified online component stage; new in-house algorithm development is paused.',size=25,color=MUTED)
text(62,1491,'Registration and scientific validation are separate; dashed links mark optional or unfinished steps.',size=22,color=MUTED)

OUT.mkdir(parents=True,exist_ok=True)
ET.ElementTree(root).write(OUT/'project_workflow.svg',encoding='utf-8',xml_declaration=True)
canvas.save(OUT/'project_workflow.png',optimize=True)
print(json.dumps({'svg':str(OUT/'project_workflow.svg'),'png':str(OUT/'project_workflow.png'),
                  'text_blocks_checked':len(boxes),'canvas':[W*SCALE,H*SCALE]},ensure_ascii=False))
