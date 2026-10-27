"""Flat SVG controls, flow arrows and compartments for PBPK Figure 3.

Every shape is explicitly drawn. No image loading, tracing or pixel masks.
"""
from __future__ import annotations
import math
from xml.etree import ElementTree as ET

NS = 'http://www.w3.org/2000/svg'
B, G, P, O, R = '#78A8BD', '#83AE99', '#A795BE', '#C9AD62', '#C77973'
NAVY = '#243642'


def path(s, d, fill='none', stroke='none', width=1.5, **kw):
    return s.el('path', d=d, fill=fill, stroke=stroke,
                stroke_width=width, stroke_linejoin='round',
                stroke_linecap='round', **kw)


def ellipse(s, cx, cy, rx, ry, fill, stroke='none', width=1.3):
    return s.el('ellipse', cx=cx, cy=cy, rx=rx, ry=ry, fill=fill,
                stroke=stroke, stroke_width=width)


def grad(s, name, stops, vertical=False):
    # Solid fills preserve the symbols without metallic or volumetric shading.
    fills = {'wrench':'#A5C9DC', 'arterial_tube':'#EAA99F',
             'tissue_1':'#F8E6DF', 'tissue_2':'#EDE6F4',
             'plasma_box':'#E0EDF3', 'slider_handle':'#A8CCBB',
             'decision_diamond':'#F7EDCD', 'repair_card':'#F5F8FA'}
    return fills.get(name, '#E7D18E')


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
    path(s,'M8 20V12C8 2 27 2 27 12V20',stroke=NAVY,width=3)
    s.rect(1,18,33,29,'#E7D18E',NAVY,5,2.6)
    ellipse(s,17.5,29,3.1,3.5,NAVY)
    path(s,'M16 31L14.7 39H20.3L19 31Z',NAVY)
    s.svg=original


def wrench(s):
    metal=grad(s,'wrench',[('0%','#E7F2FF'),('45%','#9EC7EB'),('100%','#5A92C4')],True)
    path(s,'M790 651C791 643 800 637 809 639L802 647L803 653L810 656L820 648 '
           'C821 658 816 667 804 668L787 686C784 689 779 687 776 684 '
           'C773 681 774 677 778 674L793 659Z',metal,NAVY,2.7)
    ellipse(s,782,680,2.1,2.1,'#FFFFFF',NAVY,1.5)


def mechanism(s):
    s.rect(459,98,18,145,'#EAA99F',NAVY,8,2.8)
    tissue1=grad(s,'tissue_1',[('0%','#FFF9F5'),('100%','#F7E4DE')],True)
    tissue2=grad(s,'tissue_2',[('0%','#FFFFFF'),('100%','#EEE3FC')],True)
    plasma=grad(s,'plasma_box',[('0%','#EBFAFF'),('100%','#BFE7F7')],True)
    for y,tissue,tcol in [(95,tissue1,'#8C4B3D'),(181,tissue2,'#683DA6')]:
        s.rect(501,y,67,56,plasma,NAVY,10,2.6)
        s.rect(609,y,66,56,tissue,NAVY,10,2.6)
        s.formula(r'V_{i,v}',504,y+8,61,40,size=25,color=NAVY)
        s.formula(r'V_{i,e}',611,y+8,62,40,size=25,color='#171414')
        arrow(s,478,y+20,499,y+20,B,2.2,6)
        arrow(s,498,y+36,477,y+36,B,2.2,6)
        arrow(s,570,y+19,607,y+19,B,2.8,8)
        arrow(s,608,y+36,570,y+36,B,2.8,8)
    # Flat organ symbols; these are labels for the repeated compartments.
    path(s,'M703 127C699 108 713 101 731 103L757 108Q770 110 759 121L739 135Q733 141 727 129Q716 143 705 139Z',
         '#EAA99F',NAVY,3)
    path(s,'M736 108Q738 118 730 128',stroke=NAVY,width=2)
    path(s,'M712 183C698 177 692 191 694 209Q696 230 711 229Q724 228 717 217Q706 211 714 204Q723 192 712 183Z',
         '#B7A6CA',NAVY,3)
    path(s,'M750 183C764 177 770 191 768 209Q766 230 751 229Q738 228 745 217Q756 211 748 204Q739 192 750 183Z',
         '#B7A6CA',NAVY,3)
    path(s,'M718 209H731H744M731 209V239',stroke=NAVY,width=2.8)


def controls(s):
    s.rect(1341,89,220,195,'#FFFFFF',NAVY,13,2.8)
    handle=grad(s,'slider_handle',[('0%','#A9DCCB'),('40%','#5FA98F'),('100%','#27775F')],True)
    for y,tex,hx in [(115,r's_{CL}',1490),(162,r's_K',1467),
                      (209,r's_{ex}',1490),(257,r's_{RBC}',1463)]:
        s.formula(tex,1352,y-19,58,38,size=25,color='#141617')
        path(s,f'M1414 {y}H1542',stroke='#C4D2D7',width=4)
        ellipse(s,hx,y,9.4,10.2,handle,NAVY,2.7)


def decision(s):
    diamond=grad(s,'decision_diamond',[('0%','#FFF4C3'),('100%','#FFD97A')],True)
    path(s,'M508 664L568 728L508 792L448 728Z',diamond,NAVY,3)
    s.text(508,710,'Bound\nviolated?',17,True,NAVY,'center')
    s.text(590,702,'Yes',17,True,R,'center')
    arrow(s,569,728,612,728,R,3.6,10)
    pale=grad(s,'repair_card',[('0%','#FBFDFF'),('100%','#E9F1F8')],True)
    for x in (613,768):
        s.rect(x,672,110,106,pale,NAVY,10,2.3)
    s.text(668,642,'Repair eligible?',18,True,NAVY,'center')
    lock(s,651,681,1.0)
    s.text(668,738,'Unmeasured\nRBC only',15.7,True,NAVY,'center')
    s.text(745,702,'Yes',16,True,P,'center')
    arrow(s,724,728,767,728,P,3.6,10)
    original=s.svg
    s.svg=ET.SubElement(original,'{'+NS+'}g',transform='translate(25 43)')
    wrench(s)
    s.svg=original
    s.text(823,738,'Relax RBC\nlower bound',15.7,True,NAVY,'center')
    s.text(735,800,'Otherwise',15.5,True,R,'center')
    arrow(s,668,779,668,833,R,3.6,11)
    s.rect(610,834,260,51,'#FCF0ED',R,11,1.7)
    ellipse(s,636,859,16,16,R,'white',1.4)
    path(s,'M626 859H646',stroke='white',width=4)
    s.text(752,843,'Protected or\nunresolved: stop',15.5,True,R,'center')
    s.text(486,804,'No',18.5,True,G,'center')


def flow(s, name):
    if name == 'forward':
        d='M1118 472V478Q1118 498 1098 498H420Q400 498 400 518V607'
        head='M392 607L400 620L408 607Z'; color=G
    elif name == 'refit':
        d='M823 672V545Q823 519 849 519H1222Q1246 519 1246 495V485'
        head='M1238 485L1246 472L1254 485Z'; color=R
    else:
        d='M508 794V895Q508 912 525 912H889Q906 912 906 895V708Q906 692 922 692H935'
        head='M931 684L944 692L931 700Z'; color=G
    path(s,d,stroke='white',width=7)
    path(s,d,stroke=color,width=3.8)
    path(s,head,color,color,.9)
    if name == 'pass_arrow':
        s.text(715,931,'Mass bound passed',17,True,G,'center')
    if name == 'refit':
        s.rect(1263,483,82,42,'#FFFFFF',R,10,1.6)
        s.text(1304,494,'Refit',20,True,R,'center')


def flat_ui(s, name):
    original=s.svg
    s.svg=ET.SubElement(original,'{'+NS+'}g',id='native_'+name)
    if name == 'mechanism': mechanism(s)
    elif name == 'controls': controls(s)
    elif name == 'decision': decision(s)
    elif name in ('forward','refit','pass_arrow'): flow(s,name)
    elif name == 'plus':
        path(s,'M190 177H198V185H206V193H198V202H190V193H182V185H190Z',B)
    elif name == 'input_tag':
        # One brace joins the request context to both evidence inputs.
        path(s,'M52 348V361H374V348M213 361V390',stroke=B,width=1.8)
        s.rect(44,391,359,42,'#E9F2F6','none',10)
        s.text(223.5,402,'Drug + human + dose + endpoint',20,True,NAVY,'center')
    elif name == 'renal_tag':
        s.rect(449,391,307,42,'#FAF4DE',O,10,1.7)
        lock(s,472,393,.68)
        s.text(627,402,'Qualified renal CL fixed',20,True,NAVY,'center')
        path(s,'M757 412H789',stroke=O,width=1.8)
    elif name == 'gate_tag':
        s.rect(801,112,116,80,'#FAF4DE',O,10,1.8)
        s.text(859,123,'Measured B/P:\nblock free\nRBC fit',16,True,NAVY,'center')
        path(s,'M859 193V225',stroke=O,width=1.8)
        ellipse(s,859,225,3,3,O)
    elif name == 'obs':
        s.rect(1307,843,197,49,'#FCFAFF','#7951B1',5,1.4)
        s.rect(1319,851,25,31,'white','#A696C5',1,1)
        for x,y,w,h,c in [(1324,869,4,9,'#D5B2F5'),(1330,861,4,17,'#8CAAE3'),(1336,865,4,13,'#E2C057')]:
            s.rect(x,y,w,h,c,None,0)
        path(s,'M1323 855H1337',stroke='#B8B1D3',width=1)
        s.text(1421,852,'Target observations:\nscoring only',14.2,True,NAVY,'center')
    else:
        if name == 'main_arrow':
            # A bracket attaches the model construction group to calibration.
            path(s,'M778 88H789V445H778',stroke=B,width=1.8)
        specs={
            'evidence_arrow':(393,192,441,192,B,6,14),
            'main_arrow':(789,225,944,225,B,4,13),
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
