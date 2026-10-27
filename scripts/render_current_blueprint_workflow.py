#!/usr/bin/env python3
"""Render the current paper's code-native workflow and blueprint diagrams.

Figure 1 replays the accepted renderer through panel (c), then draws a new
panel (d). No existing bitmap is edited. The historical renderer and frozen
experiment artifacts remain unchanged. Figure 2 uses the same native drawing
primitives and project-local artwork. No fitted or observed data are drawn.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from PIL import Image, ImageChops

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/figures/workflow'
OLD = OUT / 'previews/grouped_20260929'

def overview():
    path = OLD / 'render_figure_1.py'
    source = path.read_text()
    ns = {'__file__': str(path), '__name__': 'accepted_workflow_primitives'}
    # Replay source-native commands, preserving every A/B/C label and pixel.
    replay=source.split('# (d) Reflect current development calibration')[0]
    replay='\n'.join(line for line in replay.splitlines() if not line.startswith(('text(36,20,','text(39,79,')))
    exec(compile(replay, str(path), 'exec'), ns)
    C, text, lane, group, arrow, pill = (ns[k] for k in ('C', 'text', 'lane', 'group', 'arrow', 'pill'))
    P, G, B = (ns[k] for k in ('PURPLE', 'GREEN', 'BLUE'))
    ART = ns['ART']
    lane(1300, 397, 'd', 'PBPK blueprint design', 'Construct → calibrate visible data → check → lock and predict.', 3)
    for x,w,title,focus in [(58,642,'Evidence-guided construction',False),
                            (740,880,'Development calibration and checks',True),
                            (1660,682,'Locked model and assessment',False)]:
        group(x,1383,w,294,title,3,focus)
    C.asset(ART/'01_human_core.png',75,1419,120,148)
    C.asset(ART/'02_evidence_bundle.png',213,1424,189,144)
    text(204,1473,'+',33,True,P,'center')
    arrow([(405,1490),(439,1490)],P,2.5)
    for yy,s,co,fill in [(1421,'Keep evidence',G,'#E6F0E7'),(1473,'Initialize inputs',B,'#E5EFF6'),(1525,'Check scope',P,'#ECE3F4')]:
        pill(448,yy,230,s,co,27,fill=fill,height=43)
    text(372,1587,'Human template + drug-specific inputs',27,True,P,'center',609)
    text(372,1631,'Available development observations',26,True,align='center',width=609)
    for x,title,line1,line2 in [(765,'Calibrate','Visible data only','Four shared scales'),
                               (1046,'Check feasibility','Physical constraints','Protected evidence'),
                               (1327,'Repair if needed','Eligible RBC bound','Then recalibrate')]:
        C.rect(x,1425,265,140,'white',P,r=12,width=1.7)
        text(x+132.5,1435,title,27,True,P,'center',249)
        text(x+132.5,1481,line1,25,align='center',width=249)
        text(x+132.5,1523,line2,25,align='center',width=249)
    arrow([(1033,1490),(1042,1490)],P,1.4)
    arrow([(1314,1490),(1323,1490)],P,1.4)
    text(1180,1590,'Optional: assess added observations in separate analyses',26,True,P,'center',844)
    text(1180,1631,'Two partition formulas; measured inputs remain protected',25,True,align='center',width=844)
    C.icon('lock',1675,1437,71,P)
    arrow([(1746,1482),(1776,1482)],P,2.4)
    C.icon('curve',1770,1412,177,P)
    C.asset(ART/'04_blueprint_package.png',2018,1418,298,180)
    arrow([(1950,1490),(2010,1490)],P,2.4)
    text(1836,1587,'Predict target study',27,True,P,'center',306)
    text(2165,1602,'Model + endpoint report',27,True,P,'center',335)
    text(2000,1639,'Target observations enter scoring only',26,True,align='center',width=604)
    for yy,col in [(315,ns['BLUE']),(664,ns['GREEN']),(1490,P)]:
        ns['ribbon']([(708,yy),(732,yy)],col,5.4)
        ns['ribbon']([(1628,yy),(1652,yy)],col,5.4)
    ns['ribbon']([(1096,1114),(1120,1114)],ns['GOLD'],5.4)
    ns['ribbon']([(1617,1114),(1652,1114)],ns['GOLD'],5.4)
    check_source = 'def check():' + source.split('def check():',1)[1].split("if __name__=='__main__':",1)[0]
    exec(compile(check_source,str(path),'exec'),ns)
    ns['check']()
    # Validate A/B/C preservation against the accepted published graphic.
    old=Image.open(OUT/'figure_1_overview.png').convert('RGB')
    area=(0,124*C.scale,old.width,1300*C.scale)
    if ImageChops.difference(old.crop(area),C.image.crop(area)).getbbox():
        raise ValueError('Figure 1 A/B/C differ from the accepted graphic')
    # Render only the figure body; the manuscript caption supplies the title.
    # The source-native A/B/C pixels are preserved, then placed in this viewport.
    target=OUT/'figure_1_overview_current.png'
    C.image.crop((0,124*C.scale,C.w*C.scale,1703*C.scale)).save(target,dpi=(300,300))
    report={'path':str(target),'size':(C.w*C.scale,1579*C.scale),
            'preserved_panels':'a,b,c','native_text_and_geometry_qa':True}
    return report


def blueprint():
    """Compact illustrated architecture; every curve is a schematic.

    Reuse the manuscript's artwork and palette, with rounded ribbon arrows.
    Short equations support each illustrated step; detailed rules stay in text.
    """
    sys.path.insert(0, str(ROOT / 'scripts'))
    from english_workflow_art import Canvas
    C = Canvas(1400, 850, scale=3, strict_text=True)
    B, G, O, P = '#527F9E', '#4B7A65', '#A47D3D', '#8064A0'
    INK, MUTED = '#293744', '#657780'
    ART = OUT / 'blueprint_design/details'

    def txt(x, y, s, size=27, bold=False, color=INK, align='left', width=None):
        C.text(x, y, s, size, bold, color, align, max_width=width)

    def tint(color, white=.60):
        rgb = tuple(int(color[k:k+2], 16) for k in (1, 3, 5))
        return '#%02x%02x%02x' % tuple(round(v+(255-v)*white) for v in rgb)

    def flow(points, color, main=False):
        # A cubic Bezier shaft with rounded entry and a solid arrowhead.
        if len(points) == 2:
            a, b = points
            points = [a, (a[0]+(b[0]-a[0])/3, a[1]+(b[1]-a[1])/3),
                      (a[0]+2*(b[0]-a[0])/3, a[1]+2*(b[1]-a[1])/3), b]
        p0, p1, p2, p3 = points
        path = [tuple((1-t)**3*p0[k]+3*(1-t)**2*t*p1[k]
                      +3*(1-t)*t*t*p2[k]+t**3*p3[k] for k in (0,1))
                for t in [i/100 for i in range(101)]]
        half, head = (5.3, 15) if main else (2.7, 10)
        length = sum(math.dist(a,b) for a,b in zip(path,path[1:]))
        head = min(head, length*.35)
        stop, done, center = length-head, 0, [path[0]]
        for a,b in zip(path,path[1:]):
            dist = math.dist(a,b)
            if done+dist >= stop:
                f=(stop-done)/dist
                center.append((a[0]+f*(b[0]-a[0]),a[1]+f*(b[1]-a[1])))
                break
            center.append(b);done+=dist
        left, right = [], []
        for i,q in enumerate(center):
            a,b=center[max(0,i-1)],center[min(len(center)-1,i+1)]
            d=math.dist(a,b);ux,uy=(b[0]-a[0])/d,(b[1]-a[1])/d
            left.append((q[0]-uy*half,q[1]+ux*half))
            right.append((q[0]+uy*half,q[1]-ux*half))
        base=center[-1];d=math.dist(base,path[-1]);ux,uy=(path[-1][0]-base[0])/d,(path[-1][1]-base[1])/d
        shoulder=half*2.1
        l=(base[0]-uy*shoulder,base[1]+ux*shoulder)
        r=(base[0]+uy*shoulder,base[1]-ux*shoulder)
        C.polygon(left+[l,path[-1],r]+right[::-1],tint(color,.43),color,.8)
        C.circle(*center[0],half,tint(color,.43),color,.8)
        C.polygon([l,path[-1],r],color)

    def panel(x,y,color,letter,title):
        C.rect(x,y,665,376,tint(color,.94),tint(color,.30),r=17,width=1.3)
        txt(x+20,y+15,f'({letter})  {title}',31,True,color,width=625)
        C.line([(x+20,y+60),(x+645,y+60)],tint(color,.83),1)

    def axes(x,y,w,h,color=MUTED):
        C.line([(x,y),(x,y+h),(x+w,y+h)],color,1.6)

    panel(10,10,B,'a','Build from evidence')
    panel(725,10,G,'b','Calibrate visible observations')
    panel(10,458,O,'c','Check and update')
    panel(725,458,P,'d','Lock and predict')

    # A: Picture-led assembly, retaining the physiological model as the base.
    C.asset(ART/'01_human_core.png',46,91,98,159)
    C.asset(ART/'02_evidence_bundle.png',191,101,175,145)
    txt(164,157,'+',34,True,B,'center')
    txt(94,263,'Human',27,True,B,'center')
    txt(272,263,'Drug evidence',27,True,B,'center')
    flow([(375,170),(425,170)],B)
    # Vessel/tissue pairs + organ artwork: not a generic regression output.
    C.line([(443,111),(443,246)],B,3)
    for yy,organ,fill in [(106,'liver','#D5E2D9'),(174,'kidney','#E6DCF0')]:
        C.rect(454,yy,40,38,'#D6E6F0',B,r=8,width=1.3)
        C.rect(535,yy,43,38,fill,B,r=8,width=1.3)
        C.line([(494,yy+11),(531,yy+11)],B,1.5,arrow=True)
        C.line([(535,yy+29),(498,yy+29)],B,1.5,arrow=True)
        C.line([(443,yy+19),(453,yy+19)],B,1.7)
        C.icon(organ,585,yy-14,65,B)
    txt(545,263,'Human PBPK',27,True,B,'center')
    # A narrow caption ribbon carries the request and input policy.
    C.formula(342,299,r'\theta_d=\mathcal{T}(E_d;q)',size=29,
              max_width=598,color=B,align='center')
    C.rect(34,342,616,33,tint(B,.88),None,r=8)
    txt(342,345,'User endpoint · Keep evidence · Initialize inputs',26,True,B,'center',598)

    # B: The visible development set is general, not defined by a 4+1 budget.
    # Curves and six illustrative dots are schematic, never clinical values.
    axes(767,98,339,150)
    def candidate_curve(offset=0):
        return [(779+316*k/80,116+112*(1-math.exp(-3*k/80))+offset*(k/80)) for k in range(81)]
    for offset in (-17,16):C.line(candidate_curve(offset),tint(G,.70),1.5)
    base=candidate_curve();C.line(base,G,2.7)
    for k,offset in ((0,3),(11,-5),(24,4),(39,-3),(58,5),(80,-3)):
        C.circle(base[k][0],base[k][1]+offset,5.5,'white',G,1.8)
    txt(945,263,'Visible development curve',26,True,G,'center',395)
    # Four sliders illustrate the common joint parameter update. They do not
    # imply four uniquely identified physiological parameters or four samples.
    C.rect(1144,94,212,155,'white',tint(G,.38),r=12,width=1.3)
    for yy,pos,label in ((115,.64,'CL'),(150,.39,'K'),(185,.58,'EX'),(220,.32,'RBC')):
        txt(1173,yy-12,label,26,True,G,'center',50)
        C.line([(1208,yy),(1334,yy)],tint(G,.67),4)
        C.circle(1208+126*pos,yy,7,G,'white',1.1)
    flow([(1111,171),(1136,171)],G)
    txt(1248,263,'Joint update',26,True,G,'center',208)
    C.formula(1057,299,r'\hat{\mathbf{s}}=\mathrm{argmin}_{\mathbf{s}\in\mathcal{S}(E_d)}[L_{\rm dev}(\mathbf{s})+R(\mathbf{s})]',
              size=29,max_width=598,color=G,align='center')
    if C.formula_layout[-1]['bounds'][3] > 340:
        raise ValueError('Calibration equation overlaps the footer ribbon.')
    C.rect(749,342,616,33,tint(G,.88),None,r=8)
    txt(1057,345,'Two partition formulas · Evidence-constrained fit',26,True,G,'center',598)

    # C: Visible-data diagnosis protects measured inputs and bounds any repair.
    C.asset(ART/'02_evidence_bundle.png',44,548,149,119)
    C.icon('lock',43,696,44,O)
    txt(163,704,'Protected',26,True,O,'center',150)
    flow([(204,608),(243,608)],O)
    axes(263,552,148,99)
    C.polygon([(264,601),(410,601),(410,650),(264,650)],'#EDE3CA')
    C.line([(264,601),(409,601)],O,1.9,dash=True)
    C.circle(315,571,7,P,'white',1.2)
    C.line([(315,580),(315,594)],P,1.5)
    txt(334,704,'Mass check',26,True,O,'center')
    flow([(421,608),(454,608)],O)
    C.asset(ART/'05_model_execution.png',466,539,176,133)
    txt(550,704,'Refit',27,True,O,'center')
    C.formula(342,746,r'C_p(t)\leq D_{\mathrm{delivered}}(t)\max_i h_i',
              size=29,max_width=598,color=O,align='center')
    C.rect(34,793,616,33,tint(O,.87),None,r=8)
    txt(342,796,'Repair only an eligible, unmeasured RBC bound',26,True,O,'center',598)

    # D: A locked model accepts target conditions, then outputs a plasma curve.
    C.asset(ART/'04_blueprint_package.png',754,544,169,134)
    txt(839,697,'Locked model',27,True,P,'center')
    flow([(930,610),(975,610)],P)
    # Target dosing is an input; observations are never routed back to fitting.
    C.icon('bottle',1116,527,49,P)
    txt(1247,537,'Target dosing',26,True,P,'center',181)
    axes(995,579,339,98)
    curve=[]
    for k in range(101):
        t=k/100;v=(1-math.exp(-26*t))*math.exp(-3.0*t)
        curve.append((1002+324*t,675-124*v))
    C.polygon([(1002,675)]+curve+[(1326,675)],tint(P,.83))
    C.line(curve,P,3)
    peak=min(curve,key=lambda q:q[1])
    C.circle(*peak,5,P,'white',1)
    txt(peak[0]+24,577,'Cmax',26,True,P)
    txt(1080,641,'AUC',27,True,P,'center')
    txt(1168,697,'Curve + NCA endpoints',27,True,P,'center')
    C.formula(1057,746,r'\dot{\mathbf{A}}=M\mathbf{A}+u_q(t),\quad C_p=\mathbf{h}^{\mathsf{T}}\mathbf{A}',
              size=29,max_width=598,color=P,align='center')
    C.rect(749,793,616,33,tint(P,.88),None,r=8)
    txt(1057,796,'11 endpoint types · Target data: scoring only',26,True,P,'center',598)

    # Flow direction follows the reference's soft, continuous ribbon style.
    flow([(683,190),(716,190)],B,True)
    flow([(1060,395),(985,433),(428,411),(345,449)],G,True)
    flow([(683,638),(716,638)],O,True)

    overlaps=[]
    for i,a in enumerate(C.text_bounds):
        for b in C.text_bounds[i+1:]:
            if min(a[2],b[2])-max(a[0],b[0])>0 and min(a[3],b[3])-max(a[1],b[1])>0:
                overlaps.append((a[-1],b[-1]))
    if overlaps:raise ValueError('Overlapping labels: '+json.dumps(overlaps))
    if any(actual<26 for text,_,actual in C.text_sizes if text.strip()):
        raise ValueError('Diagram label below 7.35 pt at manuscript width.')
    if any(actual<29 for _,_,actual in C.formula_sizes):
        raise ValueError('Diagram equation below 8.2 pt at manuscript width.')
    report=C.save(OUT/'figure_2_blueprint_current.png')
    report.update(native_text_and_geometry_qa=True,schematic_only=True,
                  final_width_inches=5.5,final_height_inches=850*5.5/1400,
                  minimum_label_points=26*5.5*72/1400,
                  minimum_equation_points=29*5.5*72/1400,whole_figure_title=False)
    return report


if __name__ == '__main__':
    print(json.dumps({'overview':overview(),'blueprint':blueprint()},ensure_ascii=False))
