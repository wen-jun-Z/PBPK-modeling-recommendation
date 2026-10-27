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
ET.SubElement(root, SVG+'title', {'id':'title'}).text = 'PBPK 知识库与模型推荐：当前项目流程'
ET.SubElement(root, SVG+'desc', {'id':'description'}).text = (
    '证据和模型分别收集入库；用户请求分为组件推荐和公共模型推荐。'
    '当前外部实验使用证据派生模板与模型描述或图，独立科学评价仍需补齐。')
canvas = Image.new('RGB', (W*SCALE,H*SCALE), BACKGROUND)
draw = ImageDraw.Draw(canvas)
fonts, boxes = {}, []


def font(size):
    if size not in fonts:
        fonts[size] = ImageFont.truetype(str(FONT), int(size*SCALE))
    return fonts[size]


def text(x, y, value, size=25, color=INK, limit=None):
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
text(60,48,'PBPK 知识库与模型推荐',size=49)
text(62,118,'当前项目流程  /  数据底座、已有功能与科研验证',size=26,color=MUTED)
text(1535,125,'2026 · 09 · 16',size=22,color=MUTED)

section(1,'资料收集与知识库',181,TEAL)
card(60,240,300,145,'公开证据资料',['文献 · 药品标签','临床 PK 数据'],size=24)
card(430,240,530,145,'收集、抽取与质量校验',['规则抽取 + Qwen 辅助','身份归属 · 去重 · 冲突 · 溯源'],size=24)
card(1030,240,710,145,'公共证据库与检索索引',['30 种药物 · 2,872 条可用证据','保留原文位置和证据引用'],size=25)
card(60,410,300,130,'公开 PBPK 模型',['模型源码与说明'],size=24)
card(430,410,530,130,'模型获取、解析与登记',['版本 · 文件校验 · 许可证 · 结构信息'],size=24)
card(1030,410,710,130,'模型目录与结构图',['202 个模型族 · 280 个版本'],size=25)
for y in (312,475):
    polyline([(366,y),(422,y)],arrow=True)
    polyline([(966,y),(1022,y)],arrow=True)

section(2,'已有推荐功能：两个工作分支',591,BLUE)
card(60,750,300,150,'用户建模请求',['药物 · 人群 · 途径','预测任务与约束'],color=BLUE,size=23)
card(510,670,560,145,'组件推荐',['知识库检索与重排 → Qwen 生成','引用与规则校验'],color=BLUE,size=25)
card(1160,670,580,145,'有证据的组件建议',['机制、组件与参数候选','引用来源 · 冲突与不确定性'],color=BLUE,size=25)
card(510,880,560,145,'公共模型推荐',['需求画像 + 模型目录 + 机制证据','相关性与用途门控 → 分层排序'],color=BLUE,size=24)
card(1160,880,580,175,'候选模型与用途边界',['复用 / 适配 / 待验证 / 参考 / 拒绝','同药物可执行模板可进入','蓝图生成与真实运行'],color=BLUE,size=24)
polyline([(366,825),(412,825)])
polyline([(412,742),(412,952)])
polyline([(412,742),(502,742)],arrow=True)
polyline([(412,952),(502,952)],arrow=True)
polyline([(1078,742),(1152,742)],arrow=True)
polyline([(1078,952),(1152,952)],arrow=True)
polyline([(760,822),(760,871)],color=BLUE,dashed=True,arrow=True)
text(785,838,'组件证据（可选接口）',size=21,color=MUTED)

section(3,'当前模型推荐实验与下一阶段',1110,PURPLE)
card(60,1185,370,220,'冻结实验输入',['772 请求 × 280 个版本','证据派生需求模板','候选描述 / 元数据图'],color=PURPLE,size=24)
card(480,1185,370,220,'外部方法对照',['嵌入 / 重排 / 图方法','12 个配置 · 零样本','原始 / 重排 / 缺证据'],color=PURPLE,size=24)
card(900,1185,370,220,'当前可测结果',['排序变化与输出有效性','时延 / 显存 / 失败原因','保留原始及修正结果'],color=PURPLE,size=24)
card(1320,1185,420,220,'独立科学评价 · 待补齐',['先补独立标签和研究对应','再评排序、PK 误差、拒绝','与置信度校准'],color=AMBER,pending=True,size=24)
polyline([(438,1290),(472,1290)],arrow=True)
polyline([(858,1290),(892,1290)],arrow=True)
polyline([(1278,1290),(1312,1290)],color=AMBER,dashed=True,arrow=True)
text(62,1444,'当前对照使用证据派生模板，未统一串联在线组件推荐；自研新算法暂停。',size=25,color=MUTED)
text(62,1491,'模型登记与科学验证分开；虚线表示可选接口或待完成环节。',size=22,color=MUTED)

OUT.mkdir(parents=True,exist_ok=True)
ET.ElementTree(root).write(OUT/'project_workflow.svg',encoding='utf-8',xml_declaration=True)
canvas.save(OUT/'project_workflow.png',optimize=True)
print(json.dumps({'svg':str(OUT/'project_workflow.svg'),'png':str(OUT/'project_workflow.png'),
                  'text_blocks_checked':len(boxes),'canvas':[W*SCALE,H*SCALE]},ensure_ascii=False))
