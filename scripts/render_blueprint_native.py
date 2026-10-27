#!/usr/bin/env python3
"""Flat vector artwork for the PBPK blueprint workflow (manuscript Figure 3).

All illustrations, controls and arrows use authored SVG shapes, smooth Bezier
paths, solid fills and vector type. No raster image, bitmap tracing, color-run
geometry, screenshot-derived clipping mask or original-image cache is used.
"""
from __future__ import annotations
import io
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from english_workflow_art import _mathtext_backend, MATH_DIR
from blueprint_native_plots import reference_plot
from blueprint_flat_icons import flat_icon
from blueprint_flat_ui import flat_ui, arrow, path
_mathtext_backend()
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MPath
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import cairosvg
import pymupdf

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/figures/workflow'
W,H=1605,980
B,G,O,P='#78A8BD','#83AE99','#C9AD62','#A795BE'
RED='#C77973'
INK,MUTED='#243642','#64717D'
NAVY=INK
NS='http://www.w3.org/2000/svg'
XL='http://www.w3.org/1999/xlink'
ET.register_namespace('',NS)
ET.register_namespace('xlink',XL)

class Scene:
    def __init__(self):
        self.svg=ET.Element(f'{{{NS}}}svg',dict(width=str(W),height=str(H),viewBox=f'0 0 {W} {H}'))
        ET.SubElement(self.svg,f'{{{NS}}}title').text='Evidence-constrained PBPK blueprint: construction, calibration, eligible repair and locked prediction'
        ET.SubElement(self.svg,f'{{{NS}}}desc').text='Flat scientific icons with dark outlines and pastel solid fills. Transparent background. Curves are schematic. The current free four-axis fitter blocks qualified measured blood-partition evidence. The mass bound is a necessary feasibility check, not a guarantee of predictive accuracy.'
        self.bounds=[];self.nmath=0
    def el(self,tag,**kw):
        return ET.SubElement(self.svg,f'{{{NS}}}{tag}',{k.replace('_','-'):str(v) for k,v in kw.items()})
    def rect(self,x,y,w,h,fill=None,outline=None,r=8,width=1.8,dash=False):
        kw=dict(x=x,y=y,width=w,height=h,rx=r,fill=fill or 'none',stroke=outline or 'none',stroke_width=width)
        if dash:kw['stroke_dasharray']='8 7'
        self.el('rect',**kw)
    def text(self,x,y,s,size=22,bold=False,color=INK,align='left',max_width=None):
        prop=FontProperties(fname=str(MATH_DIR/('DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')))
        for j,line in enumerate(str(s).split('\n')):
            path=TextPath((0,0),line,size=size,prop=prop)
            bb=path.get_extents(); sx=.84; ww=bb.width*sx
            assert not max_width or ww<=max_width,(s,ww,max_width)
            xx=x-(ww/2 if align=='center' else ww if align=='right' else 0)-bb.x0*sx
            baseline=y+j*size*1.28+size*.85
            seg=[]
            for v,code in path.iter_segments(curves=True,simplify=False):
                op={MPath.MOVETO:'M',MPath.LINETO:'L',MPath.CURVE3:'Q',MPath.CURVE4:'C',MPath.CLOSEPOLY:'Z'}[code]
                seg.append(op+(' '.join(f'{z:.4f}' for z in v) if op!='Z' else ''))
            self.el('path',d=' '.join(seg),fill=color,transform=f'translate({xx},{baseline}) scale({sx},-1)',aria_label=line)
            self.bounds.append((xx+bb.x0*sx,baseline-bb.y1,xx+bb.x1*sx,baseline-bb.y0,line))
    def formula(self,tex,x,y,w,h=64,size=28,color=INK):
        tex=re.sub(r'\\(mathbf|mathcal|mathsf)\s+([A-Za-z])',lambda m:'\\'+m[1]+'{'+m[2]+'}',tex)
        fig=Figure(figsize=(16,2),dpi=120);FigureCanvasAgg(fig)
        obj=fig.text(.01,.5,'$'+tex+'$',fontsize=size,color=color,math_fontfamily='stix')
        fig.canvas.draw();bb=obj.get_window_extent(fig.canvas.get_renderer()).expanded(1.02,1.10)
        buf=io.BytesIO();fig.savefig(buf,format='svg',transparent=True,bbox_inches=bb.transformed(fig.dpi_scale_trans.inverted()),pad_inches=0)
        raw=buf.getvalue().decode();self.nmath+=1
        # Isolate matplotlib's repeated glyph IDs across equations.
        for id_ in re.findall(r'\bid="([^"]+)"',raw):
            new=f'e{self.nmath}_{id_}'
            raw=raw.replace('id="'+id_+'"','id="'+new+'"').replace('#'+id_+'"','#'+new+'"').replace('#'+id_+')','#'+new+')')
        root=ET.fromstring(raw);vw=float(root.attrib['width'].replace('pt',''));vh=float(root.attrib['height'].replace('pt',''))
        assert vw<=w and vh<=h,(tex,vw,vh,w,h)
        root.set('x',str(x+(w-vw)/2));root.set('y',str(y+(h-vh)/2));root.set('width',str(vw));root.set('height',str(vh))
        self.svg.append(root)
        self.bounds.append((x+(w-vw)/2,y+(h-vh)/2,x+(w+vw)/2,y+(h+vh)/2,tex))
    def reference(self, name, transform=None, omit_labels=()):
        """Draw authored geometric artwork; no screenshot data is loaded."""
        parent=self.svg
        if transform:
            self.svg=ET.SubElement(parent,f'{{{NS}}}g',transform=transform)
        if name in ('dev_curve', 'mass_curve', 'target_curve'):
            group = reference_plot(name)
            for node in list(group):
                if node.get('aria-label') in omit_labels:
                    group.remove(node)
            if name == 'mass_curve':
                # The decision connection is placed in the current layout.
                # Do not retain the original plot-local output arrow.
                for node in list(group):
                    if node.get('d') in ('M440 722H450','M449 716L459 722L449 728Z'):
                        group.remove(node)
            palette = {'#101315':INK, '#202D66':INK, '#176A53':'#679680',
                       '#5D2698':'#8E7BA8', '#A56A0C':'#A58A47',
                       '#E8D9F6':'#EDE6F4', '#C3A7DE':'#C8BBD8'}
            for node in group.iter():
                for attr in ('fill', 'stroke'):
                    if node.get(attr) in palette:
                        node.set(attr, palette[node.get(attr)])
            self.svg.append(group)
        elif name in ('human', 'evidence', 'drug_scales', 'blueprint',
                      'target_human', 'report', 'score'):
            icon=flat_icon(name)
            if name == 'drug_scales':
                for node in list(icon):
                    if node.get('d') == 'M 126 87 H 146 V 81 L 157 90 L 146 99 V 93 H 126 Z':
                        icon.remove(node)
            self.svg.append(icon)
        else:
            flat_ui(self, name)
        self.svg=parent


def draw():
    s=Scene()
    # Dashed panel outlines replace the separate title underlines.
    for x,y,w,h,title,col in [
        (20,12,908,452,'(a)  Build from evidence','#AFC9D4'),
        (944,12,641,460,'(b)  Calibrate development data','#B2CDBE'),
        (20,536,878,439,'(c)  Check feasibility','#D8C896'),
        (916,536,669,439,'(d)  Lock and predict','#C5B8D4')]:
        s.rect(x,y,w,h,None,col,15,1.6,dash=True)
        s.text(x+18,y+14,title,29,True,INK,max_width=w-36)
    # (a) Human, evidence and a repeated vascular–tissue compartment.
    s.reference('human')
    s.reference('evidence')
    s.reference('plus')
    s.reference('evidence_arrow')
    s.reference('mechanism')
    s.text(608,72,'PK-Sim-derived linear model',18,True,NAVY,'center')
    s.text(116,302,'Human template:\nvolumes + flows',19,True,NAVY,'center')
    s.text(300,302,'Drug evidence:\npartition + clearance',19,True,NAVY,'center')
    s.text(608,257,'15 organs x 2 regions + 3 blood pools',18.7,True,NAVY,'center',max_width=335)
    s.formula(r'J_i=s_{ex}Q_{i,eff}R_{BP}\left(\frac{A_{i,v}}{K_{i,v}}-\frac{A_{i,e}}{K_{i,e}}\right)',438,287,346,77,size=30)
    s.reference('input_tag')
    s.reference('renal_tag')
    s.reference('gate_tag')
    s.reference('main_arrow')
    # (b) Same curve, dashed envelope, observations and four sliders.
    s.reference('dev_curve',omit_labels=('Concentration',))
    s.formula(r'C_p',1002,86,25,27,size=20)
    s.reference('calibrate_arrow')
    s.reference('controls')
    s.text(1451,296,'Partition formulas',19,True,INK,'center')
    for x,label,col in [(1370,'1','#E8F2EC'),(1464,'2','#EEE8F4')]:
        s.rect(x,322,68,29,col,'#A5B7AD',5,1.4)
        s.text(x+34,327,label,17,True,INK,'center')
    s.text(1262,358,'Bounded calibration',21,True,INK,'center')
    s.formula(r'\hat{\mathbf s}=\arg\min_{\mathbf s\in\mathcal S}\left[L_{dev}(\mathbf s)+P(\mathbf s)\right]',1082,388,359,62,size=29)
    # Flat, single-stroke transitions preserve all original decision routes.
    s.reference('forward')
    s.reference('refit')
    # (c) Mass bound, one visible decision, eligible repair and the pass route.
    # Drug and balance illustration uses the same flat scientific icon family.
    s.text(300,593,'Visible-data check',18,True,INK,'center')
    s.rect(173,620,255,218,None,'#D9CCA9',10,1.7)
    s.reference('drug_scales',transform='translate(43 689) scale(.70) translate(-35 -657)')
    arrow(s,146,716,172,716,O,3.2,8)
    s.formula(r'C_p(t)\leq D_{delivered}(t)\max_i h_i',175,625,251,55,size=24)
    s.reference('mass_curve',transform='translate(-32 14)')
    arrow(s,428,728,448,728,O,3.2,8)
    s.reference('decision')
    s.reference('pass_arrow')
    # (d) Two independent inputs feed the same named prediction computation.
    s.text(996,593,'Locked scales +\npartition formula',18,True,INK,'center')
    s.text(1178,593,'Target human\n+ dosing',18,True,INK,'center')
    s.reference('blueprint',transform='translate(944 645) scale(.72) translate(-912 -620)')
    s.reference('target_human',transform='translate(1135 643) translate(-1115 -604)')
    s.rect(934,793,395,141,'#FAF8FC','#B5A6C7',12,1.9)
    s.text(1131.5,806,'Target prediction',20,True,INK,'center')
    s.formula(r'\frac{d\mathbf A}{dt}=M(\theta_d,\hat{\mathbf s};\mathcal H_{target})\mathbf A+\mathbf b\,r_{target}(t)',940,837,383,47,size=26)
    s.formula(r'C_p=\mathbf h_{target}^{\mathsf T}\mathbf A',1018.5,884,226,44,size=28)
    arrow(s,996,735,996,792,P,3.6,11)
    arrow(s,1178,734,1178,792,P,3.6,11)
    # Output leaves the computation and terminates on the target-curve axis.
    path(s,'M1329 857H1345Q1357 857 1357 845V697Q1357 685 1363 685',stroke=P,width=3.6)
    arrow(s,1363,685,1375,685,P,3.6,8)
    s.text(1452,594,'Predicted curve',19,True,INK,'center')
    s.reference('target_curve',transform='translate(75 0)',omit_labels=('Concentration','Time'))
    s.formula(r'C_p',1349,633,25,27,size=20)
    s.text(1524,765,'Time',18,True,INK,'right')
    # Curve/report and observed targets meet only in the scoring output.
    arrow(s,1429,752,1429,829,P,3.6,11)
    s.reference('report',transform='translate(1392 830) scale(.9) translate(-1484 -630)')
    s.text(1429,931,'Curve + NCA',17,True,INK,'center')
    s.text(1429,951,'11 output types',14.5,False,INK,'center')
    s.rect(1480,787,96,40,'#FAF8FC','#B5A6C7',7,1.7)
    s.text(1528,792,'Target\nobservations',14.5,True,INK,'center')
    arrow(s,1528,828,1528,850,P,3.2,8)
    arrow(s,1467,877,1504,877,P,3.6,10)
    s.reference('score',transform='translate(1506 852) scale(.9) translate(-1507 -826)')
    s.text(1532,931,'Scoring\nonly',16,True,INK,'center')
    # The final figure consists of authored geometry and mathematical glyphs.
    assert not list(s.svg.iter('{'+NS+'}image')), 'Raster embeds are forbidden.'
    assert not list(s.svg.iter('{'+NS+'}linearGradient')), 'Solid fills only.'
    assert not list(s.svg.iter('{'+NS+'}radialGradient')), 'Solid fills only.'
    assert not [v for v in s.bounds if min(v[0],v[1])<0 or v[2]>W or v[3]>H],s.bounds
    OUT.mkdir(parents=True,exist_ok=True)
    svg=OUT/'figure_2_blueprint_native.svg'
    png=OUT/'figure_2_blueprint_native.png'
    pdf=OUT/'figure_2_blueprint_native.pdf'
    ET.ElementTree(s.svg).write(svg,encoding='utf-8',xml_declaration=True)
    cairosvg.svg2pdf(url=str(svg),write_to=str(pdf))
    with pymupdf.open(pdf) as document:
        assert len(document)==1 and not document[0].get_images(full=True)
        scale=3210/document[0].rect.width
        document[0].get_pixmap(matrix=pymupdf.Matrix(scale,scale),alpha=True).save(png)
        preview_dir=OUT/'previews/flat_blueprint_20261026'
        preview_dir.mkdir(parents=True,exist_ok=True)
        document[0].get_pixmap(matrix=pymupdf.Matrix(scale*.65,scale*.65),alpha=False).save(preview_dir/'preview.png')
    with Image.open(png) as im:
        assert im.mode=='RGBA' and im.getchannel('A').getextrema()==(0,255)
        print('Transparent PNG:',im.size)
    print('SVG/PDF/PNG:',svg,pdf,png,'; embedded rasters: 0')

if __name__=='__main__':draw()
