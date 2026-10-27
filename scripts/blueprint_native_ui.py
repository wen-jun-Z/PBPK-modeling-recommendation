"""Authored SVG controls, flow arrows and mechanical modules for PBPK Figure 2.

Every shape is explicitly drawn. No image loading, tracing or pixel masks.
"""
from __future__ import annotations
import math
from xml.etree import ElementTree as ET

NS = 'http://www.w3.org/2000/svg'
B, G, P, O, R = '#075EA4', '#176E56', '#6D2BA6', '#A2630C', '#D31522'
NAVY = '#092A74'


def path(s, d, fill='none', stroke='none', width=1.5, **kw):
    return s.el('path', d=d, fill=fill, stroke=stroke,
                stroke_width=width, stroke_linejoin='round',
                stroke_linecap='round', **kw)


def ellipse(s, cx, cy, rx, ry, fill, stroke='none', width=1.3):
    return s.el('ellipse', cx=cx, cy=cy, rx=rx, ry=ry, fill=fill,
                stroke=stroke, stroke_width=width)


def grad(s, name, stops, vertical=False):
    defs = ET.SubElement(s.svg, '{'+NS+'}defs')
    g = ET.SubElement(defs, '{'+NS+'}linearGradient', id='native_'+name,
                      x1='0%', y1='0%', x2='0%' if vertical else '100%',
                      y2='100%' if vertical else '0%')
    for position, color in stops:
        ET.SubElement(g, '{'+NS+'}stop', offset=position, **{'stop-color':color})
    return 'url(#native_'+name+')'


def arrow(s, x1, y1, x2, y2, color, width=5, head=12, outline=True):
    a = math.atan2(y2-y1, x2-x1)
    co, si = math.cos(a), math.sin(a)
    bx, by = x2-head*co, y2-head*si
    points = [(x2,y2),(bx+head*.58*si,by-head*.58*co),
              (bx-head*.58*si,by+head*.58*co)]
    d = 'M'+'L'.join(f'{x:g} {y:g}' for x,y in points)+'Z'
    if outline:
        path(s, f'M{x1} {y1}L{bx} {by}', stroke='white', width=width+3)
        path(s, d, 'white', 'white', 2.5)
    path(s, f'M{x1} {y1}L{bx+co} {by+si}', stroke=color, width=width)
    path(s, d, color, color, .5)


def lock(s, x, y, scale=1):
    original=s.svg
    s.svg=ET.SubElement(original, '{'+NS+'}g', transform=f'translate({x} {y}) scale({scale})')
    gold=grad(s, f'lock_{int(x)}_{int(y)}', [('0%','#FFF5D8'),('45%','#FFE2A0'),('100%','#F1B85C')], True)
    path(s,'M8 20V12C8 2 27 2 27 12V20',stroke='#7A4310',width=6)
    path(s,'M8 20V12C8 2 27 2 27 12V20',stroke='#FFF6D8',width=3)
    s.rect(1,18,33,29,gold,'#84501F',5,1.6)
    path(s,'M5 22H29',stroke='#FFF9E8',width=1.7)
    ellipse(s,17.5,29,3.1,3.5,'#8F5418')
    path(s,'M16 31L14.7 39H20.3L19 31Z','#8F5418')
    s.svg=original


def wrench(s):
    metal=grad(s,'wrench',[('0%','#E7F2FF'),('45%','#9EC7EB'),('100%','#5A92C4')],True)
    path(s,'M790 651C791 643 800 637 809 639L802 647L803 653L810 656L820 648 '
           'C821 658 816 667 804 668L787 686C784 689 779 687 776 684 '
           'C773 681 774 677 778 674L793 659Z',metal,'#183D6D',1.8)
    path(s,'M778 679L798 658',stroke='#EAF7FF',width=2.4)
    ellipse(s,782,680,2.1,2.1,'#4F79A7','#173961',1)


def mechanism(s):
    from blueprint_native_anatomy import native_anatomy
    vessel=grad(s,'arterial_tube',[('0%','#940D15'),('22%','#ED2937'),('46%','#FF9A93'),('65%','#E62232'),('100%','#AB101E')])
    s.rect(459,98,18,145,vessel,'#6F1119',8,1.7)
    ellipse(s,468,99,8.6,4.1,'#ED5D5B','#6D0D17',1.5)
    ellipse(s,468,98.6,6.6,2.0,'#F9A599')
    path(s,'M463 106V232',stroke='#FFD1C3',width=1.5)
    tissue1=grad(s,'tissue_1',[('0%','#FFF9F5'),('100%','#F7E4DE')],True)
    tissue2=grad(s,'tissue_2',[('0%','#FFFFFF'),('100%','#EEE3FC')],True)
    plasma=grad(s,'plasma_box',[('0%','#EBFAFF'),('100%','#BFE7F7')],True)
    for y,tissue,tcol in [(95,tissue1,'#8C4B3D'),(181,tissue2,'#683DA6')]:
        s.rect(501,y,67,56,plasma,'#1265AF',12,1.8)
        s.rect(609,y,66,56,tissue,tcol,12,1.7)
        path(s,f'M507 {y+6}H557',stroke='white',width=1.4)
        path(s,f'M615 {y+6}H665',stroke='white',width=1.3)
        s.formula(r'V_{i,v}',504,y+8,61,40,size=25,color=NAVY)
        s.formula(r'V_{i,e}',611,y+8,62,40,size=25,color='#171414')
        arrow(s,478,y+20,499,y+20,'#145CA6',2.2,6)
        arrow(s,498,y+36,477,y+36,'#145CA6',2.2,6)
        arrow(s,570,y+19,607,y+19,'#185292',2.8,8)
        arrow(s,608,y+36,570,y+36,'#185292',2.8,8)
    s.svg.append(native_anatomy('organs'))


def controls(s):
    s.rect(1341,89,220,195,'#FFFFFF',G,15,1.7)
    path(s,'M1354 94H1547',stroke='#EBF7F1',width=2)
    handle=grad(s,'slider_handle',[('0%','#A9DCCB'),('40%','#5FA98F'),('100%','#27775F')],True)
    for y,tex,hx in [(115,r's_{CL}',1490),(162,r's_K',1467),
                      (209,r's_{ex}',1490),(257,r's_{RBC}',1463)]:
        s.formula(tex,1352,y-19,58,38,size=25,color='#141617')
        path(s,f'M1414 {y}H1542',stroke='#C1C5C5',width=6)
        path(s,f'M1414 {y-2}H1542',stroke='#D5D7D6',width=1)
        ellipse(s,hx,y,9.4,10.2,handle,'#19664D',1.7)
        path(s,f'M{hx-4} {y-5}Q{hx} {y-9} {hx+4} {y-5}',stroke='#BCE4D1',width=1.2)


def decision(s):
    diamond=grad(s,'decision_diamond',[('0%','#FFF4C3'),('100%','#FFD97A')],True)
    path(s,'M540 645L615 722L540 792L460 722Z',diamond,'#A36A16',3)
    path(s,'M540 650L609 721L540 786L466 721Z','none','#FFEAB2',1)
    s.text(539,688,'Visible\nmass bound\nviolated?',17,True,NAVY,'center')
    s.text(584,637,'Yes',19,True,R,'center')
    arrow(s,568,666,613,666,R,5,12)
    pale=grad(s,'repair_card',[('0%','#FBFDFF'),('100%','#E9F1F8')],True)
    for x,w in [(615,96),(738,125)]:
        s.rect(x,631,w,111,pale,'#1766AD',10,1.3)
    lock(s,637,638,1.03)
    s.text(663,692,'Unmeasured\nRBC only',15.7,True,NAVY,'center')
    arrow(s,713,670,737,670,P,6,12)
    wrench(s)
    s.text(800,692,'Relax eligible\nRBC lower bound',14.1,True,NAVY,'center')
    arrow(s,658,744,658,779,R,6,13)
    s.rect(631,779,190,57,'#FFF9F8','#E31C29',11,1.5)
    ellipse(s,658,807,19,19,'#FF2630','white',1.4)
    path(s,'M648 807H668',stroke='white',width=4)
    s.text(745,790,'Protected /\nunresolved: stop',15.5,True,R,'center')
    s.text(517,799,'No',18.5,True,G,'center')


def flow(s, name):
    if name == 'forward':
        d='M1118 460C1117 486 1084 489 1000 489H615C514 489 455 501 442 533'
        head='M438 521L460 535L437 547Z'; color=G
    elif name == 'refit':
        d='M826 627V545Q826 519 852 519H1158Q1246 519 1246 484'
        head='M1234 482L1246 459L1258 482Z'; color=R
    else:
        d='M539 794V847Q539 871 562 871H889'
        head='M886 858L907 871L886 884Z'; color=G
    shade=grad(s,'flow_'+name,[('0%','#7EB5A2' if color==G else '#FF8385'),
                             ('45%',color),('100%','#005239' if color==G else '#B00012')])
    path(s,d,stroke='white',width=11.2)
    path(s,head,'white','white',3)
    path(s,d,stroke=shade,width=7.8)
    path(s,d,stroke='#B2D6C7' if color==G else '#FF9294',width=1.05)
    path(s,head,shade,color,.9)
    if name == 'refit':
        s.rect(1263,483,82,42,'#FFFDFB','#EE1C25',12,1.6)
        s.text(1304,494,'Refit',20,True,R,'center')


def native_ui(s, name):
    original=s.svg
    s.svg=ET.SubElement(original,'{'+NS+'}g',id='native_'+name)
    if name == 'mechanism': mechanism(s)
    elif name == 'controls': controls(s)
    elif name == 'decision': decision(s)
    elif name in ('forward','refit','pass_arrow'): flow(s,name)
    elif name == 'plus':
        path(s,'M190 177H198V185H206V193H198V202H190V193H182V185H190Z','#1574AB')
    elif name == 'input_tag':
        s.rect(44,391,359,42,'#E2F2FC','none',14)
        s.text(223.5,402,'Drug + human + dose + endpoint',20,True,'#0758A6','center')
    elif name == 'renal_tag':
        s.rect(449,400,307,45,'#FFF6DD','#9E5B04',16,2)
        lock(s,472,404,.68)
        s.text(627,412,'Qualified renal CL fixed',20,True,'#874500','center')
    elif name == 'gate_tag':
        s.rect(801,112,116,80,'#FFF6E5','#A26818',16,1.8)
        s.text(859,123,'Measured B/P:\nblock free\nRBC fit',16,True,'#1E211F','center')
    elif name == 'obs':
        s.rect(1307,843,197,49,'#FCFAFF','#7951B1',5,1.4)
        s.rect(1319,851,25,31,'white','#A696C5',1,1)
        for x,y,w,h,c in [(1324,869,4,9,'#D5B2F5'),(1330,861,4,17,'#8CAAE3'),(1336,865,4,13,'#E2C057')]:
            s.rect(x,y,w,h,c,None,0)
        path(s,'M1323 855H1337',stroke='#B8B1D3',width=1)
        s.text(1421,852,'Target observations:\nscoring only',14.2,True,NAVY,'center')
    else:
        specs={
            'evidence_arrow':(393,192,441,192,B,6,14),
            'main_arrow':(804,225,916,225,'#075CAB',9,20),
            'calibrate_arrow':(1285,198,1329,198,G,5,14),
            'lock_arrow':(1062,658,1105,658,P,5,14),
            'target_down':(1155,735,1155,782,P,5,13),
            'target_input_arrow':(1217,747,1257,747,P,5,13),
            'curve_report_arrow':(1456,703,1481,703,P,5,13),
            'report_down':(1530,793,1530,824,P,5,13),
        }
        if name not in specs: raise ValueError(name)
        arrow(s,*specs[name])
    s.svg=original
