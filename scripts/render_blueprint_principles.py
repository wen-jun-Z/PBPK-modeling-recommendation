"""Render the detailed four-panel PBPK blueprint principles preview.

Illustrations are independent assets. Native equations/labels describe audited
method logic; native model and prediction illustrations are schematic.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from english_workflow_art import Canvas

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/figures/workflow/blueprint_design'
DETAILS=OUT/'details'
W,H=2040,1390
C=Canvas(W,H,scale=2,strict_text=True)
INK,MUTED='#17232C','#4D606B'
BLUE,GREEN,GOLD,PURPLE='#527F9E','#4B7A65','#A47D3D','#8064A0'
A_BG,B_BG,C_BG,D_BG='#F5F9FC','#F3F8F4','#F8FAF6','#F7F5FB'
CONNECTORS=[]
CONNECTOR_MASKS=[]


def txt(x,y,s,size=26,bold=False,color=INK,align='left',width=None):
    C.text(x,y,s,size=size,bold=bold,color=color,align=align,max_width=width)


def eq(x,y,s,size=30,width=None):
    return C.formula(x,y,s,size=size,color=INK,align='center',max_width=width)


def tint(color,amount):
    rgb=[int(color[i:i+2],16) for i in (1,3,5)]
    return '#'+''.join(f'{round(v+(255-v)*amount):02x}' for v in rgb)


def unit(a,b):
    dx,dy=b[0]-a[0],b[1]-a[1]
    length=math.hypot(dx,dy)
    return (dx/length,dy/length) if length else (1.,0.)


def sample_line(a,b):
    n=max(2,math.ceil(math.dist(a,b)/2))
    return [(a[0]+(b[0]-a[0])*i/n,a[1]+(b[1]-a[1])*i/n) for i in range(n+1)]


def ribbon(path,color,main=False):
    """Rounded filled shaft with an outlined shoulder and solid arrowhead."""
    half=5.4 if main else 2.25
    head_half=11 if main else 5.5
    head_length=17 if main else 8
    length=sum(math.dist(a,b) for a,b in zip(path,path[1:]))
    head_length=min(head_length,length*.38)
    stop=length-head_length
    center=[path[0]]
    done=0
    for a,b in zip(path,path[1:]):
        segment=math.dist(a,b)
        if done+segment>=stop:
            q=(stop-done)/segment
            center.append((a[0]+q*(b[0]-a[0]),a[1]+q*(b[1]-a[1])))
            break
        center.append(b)
        done+=segment
    left,right=[],[]
    for i,p in enumerate(center):
        ux,uy=unit(center[max(0,i-1)],center[min(len(center)-1,i+1)])
        left.append((p[0]-uy*half,p[1]+ux*half))
        right.append((p[0]+uy*half,p[1]-ux*half))
    base=center[-1]
    ux,uy=unit(base,path[-1])
    shoulder_l=(base[0]-uy*head_half,base[1]+ux*head_half)
    shoulder_r=(base[0]+uy*head_half,base[1]-ux*head_half)
    ux0,uy0=unit(center[0],center[1])
    angle=math.atan2(uy0,ux0)
    cap=[(center[0][0]+half*math.cos(angle-math.pi/2-math.pi*i/12),
          center[0][1]+half*math.sin(angle-math.pi/2-math.pi*i/12)) for i in range(1,13)]
    polygon=left+[shoulder_l,path[-1],shoulder_r]+list(reversed(right))+cap
    C.polygon(polygon,tint(color,.56 if main else .25),color,1.2 if main else .8)
    C.polygon([shoulder_l,path[-1],shoulder_r],color)
    CONNECTORS.append({'path':path,'radius':half+1,'style':'main ribbon' if main else 'short filled arrow'})


def arrow(a,b,color=BLUE,width=2.6,dash=False,main=False):
    ribbon(sample_line(a,b),color,main)


def bezier(points):
    p0,p1,p2,p3=points
    path=[]
    for i in range(61):
        t=i/60
        path.append(tuple((1-t)**3*p0[k]+3*(1-t)**2*t*p1[k]+3*(1-t)*t*t*p2[k]+t**3*p3[k] for k in (0,1)))
    return path


def curve(points,color=BLUE,width=2.5,head=True,main=False):
    path=bezier(points)
    if head:
        ribbon(path,color,main)
    else:
        C.line(path,color,width)


def dashed_path(path,color,width,on=13,off=8):
    """Carry dash phase over curve samples, avoiding per-segment restarts."""
    phase=0.
    for a,b in zip(path,path[1:]):
        length=math.dist(a,b)
        if not length:
            continue
        start=0.
        while start<length-1e-8:
            draw=phase<on
            step=min(length-start,(on if draw else on+off)-phase)
            p=(a[0]+(b[0]-a[0])*start/length,a[1]+(b[1]-a[1])*start/length)
            q=(a[0]+(b[0]-a[0])*(start+step)/length,a[1]+(b[1]-a[1])*(start+step)/length)
            if draw and step>0:
                C.line([p,q],color,width)
                C.circle(*p,width/2,color)
                C.circle(*q,width/2,color)
            start+=step
            phase=(phase+step)%(on+off)


def feedback():
    path=bezier([(943,1164),(969,1190),(966,1264),(887,1264)])
    path+=sample_line((887,1264),(312,1264))[1:]
    path+=bezier([(312,1264),(69,1264),(56,1184),(89,1041)])[1:]
    C.line(path,tint(GREEN,.74),7)
    dashed_path(path,GREEN,2.2)
    C.circle(*path[0],4.2,'white',GREEN,1.6)
    ux,uy=unit(path[-4],path[-1])
    x,y=path[-1]
    C.polygon([(x,y),(x-12*ux-6*uy,y-12*uy+6*ux),
               (x-12*ux+6*uy,y-12*uy-6*ux)],GREEN)
    CONNECTORS.append({'path':path,'radius':4.2,'style':'conditional feedback'})


def panel(x,y,label,title,color,fill):
    C.rect(x,y,952,552,fill,'#C3D3DB',r=19,width=1.6)
    txt(x+28,y+20,label,34,True,color)
    txt(x+87,y+22,title,32,True,width=816)
    C.line([(x+29,y+75),(x+923,y+75)],'#D2DEE1',1)


def pill(x,y,w,label,color,size=23,h=38,fill='white'):
    C.rect(x,y,w,h,fill,color,r=11,width=1.4)
    txt(x+w/2,y+(h-size*1.16)/2-1,label,size,True,color,'center',w-16)


def role(x,y,kind,label,color):
    C.rect(x,y,106,65,'white',color,r=10,width=1.4)
    eq(x+53,y+9,r'D_{\mathrm{'+kind+'}}',25)
    txt(x+53,y+37,label,19,True,color,'center')


def switch(x,y,on):
    C.rect(x,y,61,29,'#CDE2D5' if on else '#E6E9EC',GREEN if on else '#8A989F',r=15,width=1.5)
    C.circle(x+(45 if on else 16),y+14.5,10,GREEN if on else '#8A989F')


def node(x,y,label,fill=GREEN):
    C.rect(x,y,73,64,'white',fill,r=10,width=1.6)
    eq(x+36,y+8,label,26)
    C.rect(x+20,y+42,12,10,'#C2D9C9',r=2)
    C.rect(x+40,y+42,12,10,'#BFD4E7',r=2)


def sensitivity(x,y):
    # Glyph only: two nearly collinear columns, not measured sensitivities.
    shades=['#E5EEE6','#C5D9C9','#90B599','#5F8D6B']
    for i,v in enumerate([0,1,2,3]):
        for j in range(2):
            C.rect(x+j*32,y+i*25,27,20,shades[v],r=3)
    arrow((x+76,y+47),(x+110,y+47),GREEN,2)
    txt(x+92,y+9,'SVD',18,True,GREEN,'center')
    C.line([(x+123,y+98),(x+204,y+98)],GREEN,1.5)
    C.rect(x+135,y+24,22,74,GREEN,r=2)
    C.rect(x+172,y+90,22,8,'#B5CBB9',r=2)


txt(36,22,'Evidence-driven PBPK blueprint construction',46,True)
txt(39,79,'Evidence binding → deterministic assembly → native execution → endpoint prediction',26,color=MUTED)
panel(32,134,'(a)','Ground the construction',BLUE,A_BG)
panel(1056,134,'(b)','Assemble the drug module',GREEN,B_BG)
panel(32,754,'(c)','Execute and audit the native model',GREEN,C_BG)
panel(1056,754,'(d)','Predict and compare endpoints',PURPLE,D_BG)

# A — new construction uses physiology, curated drug evidence and a declared request.
for x,label in [(199,'Human core'),(464,'Drug evidence'),(817,'Modeling goal')]:
    txt(x,237,label,27,True,align='center')
C.asset(DETAILS/'01_human_core.png',115,282,171,194)
C.icon('lock',260,285,42,BLUE)
C.asset(DETAILS/'02_evidence_bundle.png',345,292,239,166)
txt(317,354,'+',35,True,BLUE,'center')
txt(653,354,'+',35,True,BLUE,'center')
C.rect(710,289,216,157,'white',BLUE,r=15,width=1.5)
for y,label in [(309,'Adult individual'),(354,'IV dose + timing'),(399,'Plasma parent')]:
    txt(818,y,label,25,y==309,INK,'center')
for x in (199,464,817):
    C.line([(x,478),(x,491)],BLUE,1.7)
C.line([(199,491),(817,491)],BLUE,1.7)
for x in (199,464,817):
    C.circle(x,478,2.8,'white',BLUE,1.3)
arrow((314,491),(314,521),BLUE,2)
arrow((758,491),(758,521),BLUE,2)
txt(301,526,'Bind source records',25,True,BLUE,'center')
txt(761,526,'Check evidence context',25,True,BLUE,'center')
C.rect(76,568,449,83,'white','#AFC5D6',r=11,width=1.4)
txt(98,578,'Value · unit · source · status',24,True,width=405)
txt(98,615,'Measured / derived / predicted / assumed',21,color=MUTED,width=405)
for x,label in [(563,'Identity'),(691,'Units'),(819,'Scope')]:
    pill(x,585,112,label,BLUE,22,h=43)
txt(749,641,'Missing evidence → gap draft',21,True,BLUE,'center',393)

# B — deterministic mapping, explicit elimination allocation and prior partition choice.
for x,label,w in [(1220,'Partition rules',290),(1534,'Drug properties',282),(1845,'Clearance inputs',281)]:
    txt(x,237,label,27,True,align='center',width=w)
C.asset(DETAILS/'03_mechanism_modules.png',1080,282,279,162)
pill(1088,463,272,'PK-Sim Standard',GREEN,22)
pill(1088,509,272,'Rodgers–Rowland',GREEN,22)
txt(1225,558,'Declare before prediction',21,True,GREEN,'center',276)
C.rect(1402,293,267,186,'white','#9EBBAB',r=15,width=1.5)
for yy,label,symbol in [(309,'Binding',r'f_u'),(363,'Lipophilicity',r'\log P'),(417,'Ionization',r'\mathrm{p}K_a')]:
    txt(1420,yy,label,24,True)
    eq(1626,yy,symbol,26,width=65)
eq(1534,514,r'\theta_d=\mathcal{T}(E_d;q)',29,width=294)
txt(1534,558,'Typed evidence mapping',21,True,GREEN,'center',278)
C.icon('liver',1727,294,79,GREEN)
C.icon('kidney',1870,294,79,GREEN)
eq(1767,385,r'CL_H',26,width=112)
eq(1910,385,r'CL_R',26,width=112)
eq(1845,446,r'CL_H=CL_{\mathrm{total}}-CL_R',27,width=290)
txt(1845,492,'If total + renal CL are given',21,True,GREEN,'center',284)
txt(1845,529,'Do not add total CL twice',21,color=MUTED,align='center',width=282)
C.line([(1090,593),(1971,593)],'#C4D7CC',1)
for x,w,label in [(1104,247,'Individual'),(1420,247,'Compound'),(1736,247,'Protocol')]:
    pill(x,613,w,label,GREEN,24,h=44)
arrow((1363,635),(1405,635),GREEN,2.5)
arrow((1679,635),(1720,635),GREEN,2.5)

# C — no fitting or sensitivity search is performed on the current construction path.
for x,label,w in [(214,'Native snapshot',293),(506,'Native execution',284),(808,'Readback audit',292)]:
    txt(x,853,label,26,True,align='center',width=w)
C.rect(89,911,245,181,'white',BLUE,r=16,width=1.8)
txt(211,934,'< PBPK model >',28,True,BLUE,'center')
for yy,label in [(983,'Human + drug'),(1029,'Dose + output')]:
    txt(211,yy,label,24,align='center')
arrow((346,999),(378,999),GREEN,2.5)
C.rect(388,911,244,181,'white',GREEN,r=16,width=1.8)
txt(510,933,'ODE solver',29,True,GREEN,'center')
txt(510,981,'CPU · CVODES',25,True,align='center')
txt(510,1033,'One parameter set',23,align='center')
arrow((644,999),(684,999),GREEN,2.5)
for yy,label in [(916,'Dosing settings'),(972,'Drug parameters'),(1028,'Clearance capacity')]:
    C.rect(698,yy,246,44,'white','#97B4A4',r=11,width=1.4)
    C.circle(718,yy+22,5.7,GREEN)
    txt(735,yy+7,label,23,True,GREEN,width=198)
txt(211,1118,'Snapshot + executable files',22,color=MUTED,align='center',width=290)
eq(510,1111,r'\dot A=f_{\mathcal{H}}(A,\theta_d,u_q)',29,width=304)
txt(814,1118,'Check native input mapping',22,color=MUTED,align='center',width=295)
arrow((510,1158),(510,1190),GREEN,2.5)
C.rect(98,1203,824,77,'#E7F0E9','#A8C2B1',r=16,width=1.5)
C.icon('lock',121,1216,44,GREEN)
txt(194,1218,'Persist one plasma trajectory',27,True,GREEN,width=391)
eq(753,1225,r'\widehat C_p(t)=h_q(A(t))',29,width=296)

# D — all endpoints use one locked trajectory; historical cases are development data.
txt(1252,853,'Trajectory-derived endpoints',26,True,align='center',width=362)
txt(1740,853,'Endpoint prediction report',27,True,align='center',width=484)
for yy,title,label in [(913,'Exposure',r'\mathrm{AUC}\ \cdot\ \mathrm{AUC}_{\infty}'),
                       (990,'Peak + timing',r'C_{\max}\ \cdot\ t_{\max}\ \cdot\ C_{\mathrm{last}}'),
                       (1067,'Terminal kinetics',r't_{1/2}\ \cdot\ CL\ \cdot\ V_z\ \cdot\ \mathrm{MRT}\ \cdot\ V_{\mathrm{ss}}')]:
    C.rect(1096,yy,327,66,'white','#B4A6C7',r=12,width=1.5)
    txt(1260,yy+7,title,24,True,PURPLE,'center',304)
    eq(1260,yy+36,label,24,width=307)
C.asset(DETAILS/'05_model_execution.png',1456,903,486,258)
arrow((1429,1024),(1460,1024),PURPLE,2.6)
txt(1735,1170,'Schematic outputs · no measured values',21,color=MUTED,align='center',width=499)
txt(1260,1151,'Fixed terminal quality gates',22,True,PURPLE,'center',width=342)
txt(1260,1180,'Failed gates → not estimable',20,color=MUTED,align='center',width=342)
C.line([(1092,1207),(1970,1207)],'#D6CCE3',1)
C.icon('lock',1100,1231,42,PURPLE)
txt(1211,1230,'Lock',24,True,PURPLE,'center')
txt(1211,1258,'predictions',21,align='center')
arrow((1286,1253),(1320,1253),PURPLE,2.3)
txt(1452,1230,'Match time + window',23,True,PURPLE,'center',247)
txt(1452,1260,'Historical observations',21,align='center',width=256)
arrow((1591,1253),(1624,1253),PURPLE,2.3)
txt(1795,1230,'Compare by endpoint',24,True,PURPLE,'center',321)
txt(1795,1260,'Report errors + unavailable scores',21,align='center',width=335)

# Filled ribbon bridges retain the accepted four-panel visual style.
arrow((995,393),(1045,393),BLUE,main=True)
curve([(1187,697),(1091,735),(984,706),(950,744)],GREEN,head=True,main=True)
arrow((995,1008),(1045,1008),PURPLE,main=True)
C.line([(40,1340),(2000,1340)],'#C9D7DE',1)
txt(40,1354,'Scope shown: adult, single-dose IV · plasma parent drug.',20,color=MUTED)
txt(2000,1354,'Execution integrity and prediction accuracy are assessed separately.',20,color=MUTED,align='right')


def audit():
    overlaps=[]
    for i,a in enumerate(C.text_bounds):
        for b in C.text_bounds[i+1:]:
            if min(a[2],b[2])-max(a[0],b[0])>2 and min(a[3],b[3])-max(a[1],b[1])>2:
                overlaps.append([a[-1],b[-1]])
    for connector in CONNECTORS:
        r=connector['radius']
        for x,y in connector['path']:
            if any(l-r<=x<=rr+r and t-r<=y<=b+r for l,t,rr,b in CONNECTOR_MASKS):
                continue
            for l,t,rr,b,label in C.text_bounds:
                if l-r<x<rr+r and t-r<y<b+r:
                    finding=[connector['style'],label]
                    if finding not in overlaps:
                        overlaps.append(finding)
    return overlaps


if __name__=='__main__':
    overlaps=audit()
    if overlaps:
        raise RuntimeError(json.dumps(overlaps,ensure_ascii=False,indent=2))
    report=C.save(OUT/'preview.png')
    p=OUT/'prompts.json'
    meta=json.loads(p.read_text())
    meta['diagram'].update({
        'title':'Evidence-driven PBPK blueprint construction',
        'layout':'Four panels: evidence binding, deterministic assembly, native execution/readback, endpoint prediction and historical comparison.',
        'source':'scripts/render_blueprint_principles.py',
        'active_assets':['01_human_core.png','02_evidence_bundle.png','03_mechanism_modules.png','05_model_execution.png'],
        'notation':{'mathcal_H':'registered human physiology and native equations','E_d':'curated drug evidence',
                    'q':'declared request','theta_d':'evidence-bound drug parameters','T':'deterministic typed mapping',
                    'A':'amount states','u_q':'dosing input','h_q':'requested observation mapping',
                    'C_p_hat':'predicted plasma concentration','CL_H':'hepatic clearance input','CL_R':'renal clearance input',
                    'CL_total':'total clearance input; subtraction applies when total and renal inputs are supplied',
                    'CL':'single-dose IV clearance derived from the concentration trajectory; not guaranteed equal to the native clearance inputs'},
        'scope_shown':'Adult single-drug, single-dose intravenous administration; total parent-drug plasma concentration. Oral apparent endpoints are not shown.',
        'failure_paths':'Missing required evidence yields a gap draft; failed terminal gates leave affected predictions not estimable. Missing comparable observations leave scores unavailable.',
        'illustration_notice':'Conceptual illustrations and schematic prediction graphics; no measured outcomes or independent-validation claim. Current evidence assembly does not fit parameters or search mechanisms.',
        'arrow_design':'Filled ribbon bridges and rounded filled short arrows; no calibration feedback loop.',
        'design_basis':['src/pbpk_kb/blueprints/evidence_blueprint.py','src/pbpk_kb/blueprints/evidence_native_audit.py',
                        'src/pbpk_kb/blueprints/endpoint_prediction.py','src/pbpk_kb/blueprints/terminal_pk.py',
                        'experiments/blueprint_design/ten_drug_endpoints_20260923/final_results/README.md'],
        'render_audit':{'canvas':[W,H],'scale':2,'text_collisions':0,'connector_text_collisions':0,'auto_font_shrink':False},
    })
    p.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))
