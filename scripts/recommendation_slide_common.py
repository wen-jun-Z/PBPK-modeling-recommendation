"""Shared editable diagram primitives matching the existing blueprint deck."""
from build_blueprint_method_slides import *


def recommendation_slide(title, subtitle, section, notes, *, module):
    slide=Slide(title,subtitle,section,notes)
    slide.ops[0]['rows']=[f'PBPK {module}   /   {section}']
    label='Component recommendation' if module=='COMPONENTS' else 'Model recommendation'
    slide.ops[4]['rows']=[f'{label} · Principles and workflow · 2026.10.12']
    return slide


def box(s,x,y,w,h,title,detail=None,*,color=TEAL,fill=PALE,size=17):
    s.rect(x,y,w,h,fill,color,.9)
    title_y=y+10 if detail else y+(h-size*1.4)/2
    used=s.text(x+10,title_y,w-20,title,size,color,align='center',line=1.25)
    if detail:
        bottom=s.text(x+10,y+used+15,w-20,detail,13,MUTED,align='center',line=1.25)
        if used+15+bottom>h:raise ValueError('Diagram card overflow: '+title)


def arrow(s,x1,y1,x2,y2,*,color=TEAL,dash=False):
    s.line([(x1,y1),(x2,y2)],color,1.6,arrow=True,dash=dash)


def number_card(s,x,y,w,value,label,*,color=TEAL,fill=PALE):
    s.rect(x,y,w,92,fill,None)
    s.text(x+8,y+9,w-16,value,30,color,align='center')
    s.text(x+8,y+56,w-16,label,14,color,align='center',line=1.2)


def labelled_node(s,x,y,w,label,*,color=TEAL,fill=PALE):
    box(s,x,y,w,42,label,color=color,fill=fill,size=15)


def mini_matrix(s,x,y,rows=4,cols=7,color=TEAL):
    for i in range(rows):
        for j in range(cols):
            s.rect(x+j*11,y+i*11,8,8,color if (i+j*2)%5<2 else 'E7EFF2',None,rounded=False)
