"""Project-local, CPU-only illustrated English methods figure drawing helpers.

All labels and cartoon illustrations are drawn from primitives. Equations use
MathText with a project-only font registry; no system font scan occurs.
"""
from __future__ import annotations

import math
import re
import os
import json
from pathlib import Path
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT_DIR = ROOT / 'runtime/fonts/comic_neue'
MATH_DIR = ROOT / 'runtime/app-env/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf'
INK = '#293744'


@lru_cache(maxsize=128)
def _font(size: int, bold=False, math_font=False):
    path = (MATH_DIR / ('STIXGeneralBol.ttf' if bold else 'STIXGeneral.ttf')) if math_font else (FONT_DIR / ('ComicNeue-Bold.ttf' if bold else 'ComicNeue-Regular.ttf'))
    return ImageFont.truetype(str(path), max(1, int(size)))


def _color(value):
    return value


@lru_cache(maxsize=1024)
def _text_font(size,bold,char):
    """Use the local math face for symbols absent from Comic Neue."""
    f=_font(size,bold)
    if ord(char)>127 and bytes(f.getmask(char))==bytes(f.getmask('\uffff')):
        return _font(size,bold,True)
    return f


def _runs(text,size,bold):
    runs=[]
    for ch in text:
        f=_text_font(size,bold,ch)
        if runs and runs[-1][1] is f:runs[-1]=(runs[-1][0]+ch,f)
        else:runs.append((ch,f))
    return runs


def _text_width(text,size,bold):
    return sum(f.getlength(t) for t,f in _runs(text,size,bold))


@lru_cache(maxsize=1)
def _mathtext_backend():
    """Bootstrap Matplotlib from an explicit project-only font cache.

    FontManager's normal constructor discovers system fonts. A valid empty
    serialized manager bypasses that constructor; only bundled project fonts
    are then registered. No system font discovery is needed or permitted.
    """
    cache=ROOT/'runtime/cache/workflow_mathtext'
    cache.mkdir(parents=True,exist_ok=True)
    source=MATH_DIR.parents[2]/'font_manager.py'
    version=re.search(r"__version__\s*=\s*['\"]([^'\"]+)",source.read_text()).group(1)
    payload={'_version':version,'_FontManager__default_weight':'normal','default_size':None,'defaultFamily':{'ttf':'DejaVu Sans','afm':'Helvetica'},'afmlist':[],'ttflist':[],'__class__':'FontManager'}
    # Each renderer process starts from the same explicit cache; font entries
    # remain in memory. Atomic writes also allow the two figures in parallel.
    target=cache/f'fontlist-v{version}.json'
    tmp=cache/f'fontlist-{os.getpid()}.tmp'
    tmp.write_text(json.dumps(payload));os.replace(tmp,target)
    rc=cache/'matplotlibrc'
    rc.write_text('backend: Agg\nmathtext.fontset: stix\nfont.family: STIXGeneral\n')
    os.environ['MPLCONFIGDIR']=str(cache)
    os.environ['MATPLOTLIBRC']=str(rc)
    import matplotlib
    from matplotlib import font_manager as fm
    def no_scan(*args,**kwargs):
        raise RuntimeError('System font scanning is disabled for project figures.')
    fm.findSystemFonts=no_scan
    for font_path in sorted(MATH_DIR.glob('*.ttf')):
        fm.fontManager.addfont(str(font_path))
    for font_path in sorted(FONT_DIR.glob('*.ttf')):
        fm.fontManager.addfont(str(font_path))
    # Reject any accidental non-project font entry already imported by callers.
    for entry in fm.fontManager.ttflist:
        if not Path(entry.fname).resolve().is_relative_to(ROOT):
            raise RuntimeError('Non-project font found in figure font registry.')
    from matplotlib.mathtext import MathTextParser
    return MathTextParser('agg'),fm.FontProperties


class Canvas:
    def __init__(self, w=2400, h=1500, scale=2, strict_text=False):
        self.w, self.h, self.scale = w, h, scale
        self.image = Image.new('RGB', (int(w*scale), int(h*scale)), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.text_bounds = []
        self.strict_text = strict_text
        self.text_sizes = []
        self.formula_sizes = []
        self.formula_layout = []

    def text_width(self, value, size, bold=False):
        """Logical width at the requested size; layout code controls wrapping."""
        return max((_text_width(line, round(size*self.scale), bold)/self.scale
                    for line in str(value).split('\n')), default=0)

    def _xy(self, p):
        return tuple(int(round(v*self.scale)) for v in p)

    def asset(self, path, x, y, w, h):
        """Place a project-owned transparent illustration without distorting it."""
        path=Path(path).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Illustrations must be saved inside the project.')
        if min(x,y)<0 or min(w,h)<=0 or x+w>self.w or y+h>self.h:
            raise ValueError('Illustration placement exceeds the canvas.')
        with Image.open(path) as source:
            art=source.convert('RGBA')
        bounds=art.getchannel('A').getbbox()
        if bounds is None:
            raise ValueError(f'Illustration has no visible content: {path.name}')
        art=art.crop(bounds)
        ratio=min(w*self.scale/art.width,h*self.scale/art.height)
        art=art.resize((max(1,round(art.width*ratio)),max(1,round(art.height*ratio))),Image.Resampling.LANCZOS)
        px=round((x+w/2)*self.scale-art.width/2)
        py=round((y+h/2)*self.scale-art.height/2)
        self.image.paste(art,(px,py),art)

    def rect(self, x, y, w, h, fill, outline=None, r=12, width=2):
        self.draw.rounded_rectangle(self._xy((x,y,x+w,y+h)), radius=int(r*self.scale), fill=fill, outline=outline, width=max(1,int(width*self.scale)))

    def circle(self, cx, cy, r, fill, outline=None, width=2):
        self.draw.ellipse(self._xy((cx-r,cy-r,cx+r,cy+r)),fill=fill,outline=outline,width=max(1,int(width*self.scale)))

    def ellipse(self,x,y,w,h,fill,outline=None,width=2):
        self.draw.ellipse(self._xy((x,y,x+w,y+h)),fill=fill,outline=outline,width=max(1,int(width*self.scale)))

    def polygon(self,points,fill,outline=None,width=2):
        pts=[self._xy(p) for p in points]
        self.draw.polygon(pts,fill=fill)
        if outline:self.draw.line(pts+[pts[0]],fill=outline,width=max(1,int(width*self.scale)),joint='curve')

    def line(self,points,fill='#36434b',width=2,arrow=False,dash=False):
        if len(points)<2:return
        pts=[self._xy(p) for p in points]
        if dash:
            for a,b in zip(pts,pts[1:]):
                dx,dy=b[0]-a[0],b[1]-a[1]; length=math.hypot(dx,dy)
                if not length:continue
                d=0
                while d<length:
                    e=min(d+7*self.scale,length)
                    self.draw.line([(a[0]+dx*d/length,a[1]+dy*d/length),(a[0]+dx*e/length,a[1]+dy*e/length)],fill=fill,width=max(1,int(width*self.scale)))
                    d+=12*self.scale
        else:self.draw.line(pts,fill=fill,width=max(1,int(width*self.scale)),joint='curve')
        if arrow:
            a,b=points[-2:];dx,dy=b[0]-a[0],b[1]-a[1];l=math.hypot(dx,dy)
            if l:
                ux,uy=dx/l,dy/l;sz=max(8,4*width)
                self.polygon([b,(b[0]-sz*ux+sz*.43*uy,b[1]-sz*uy-sz*.43*ux),(b[0]-sz*ux-sz*.43*uy,b[1]-sz*uy+sz*.43*ux)],fill)

    def text(self,x,y,text,size=26,bold=False,color=INK,align='left',max_width=None):
        text=str(text)
        lines=text.split('\n');fs=float(size)
        if self.strict_text and max_width and self.text_width(text, size, bold)>max_width:
            raise ValueError(f'Text needs wrapping or more space at {size}px: {text!r}; '
                             f'width={self.text_width(text,size,bold):.1f}, available={max_width}')
        while max_width and max(_text_width(line,round(fs*self.scale),bold) for line in lines)>max_width*self.scale and fs>9:
            fs-=.5
        self.text_sizes.append((text, size, fs))
        f=_font(round(fs*self.scale),bold)
        lh=fs*1.16
        for i,line in enumerate(lines):
            ww=_text_width(line,round(fs*self.scale),bold)/self.scale
            xx=x-ww/2 if align=='center' else x-ww if align=='right' else x
            yy=y+i*lh
            cursor=xx*self.scale
            # Align all fallback runs on the Comic Neue baseline; the final
            # bbox remains governed by the explicitly requested label size.
            baseline=yy*self.scale+f.getmetrics()[0]
            boxes=[]
            for run,rf in _runs(line,round(fs*self.scale),bold):
                self.draw.text((round(cursor),round(baseline)),run,font=rf,fill=color,anchor='ls')
                boxes.append(self.draw.textbbox((round(cursor),round(baseline)),run,font=rf,anchor='ls'))
                cursor+=rf.getlength(run)
            if boxes:self.text_bounds.append((min(b[0] for b in boxes)/self.scale,min(b[1] for b in boxes)/self.scale,max(b[2] for b in boxes)/self.scale,max(b[3] for b in boxes)/self.scale,line))
        return len(lines)*lh

    def formula(self,x,y,tex,size=26,max_width=None,color=INK,align='left',valign='top'):
        parser,FontProperties=_mathtext_backend()
        tex=re.sub(r'\\(mathsf|mathrm|mathbf|mathit)\s+([A-Za-z0-9])',lambda m:'\\'+m.group(1)+'{'+m.group(2)+'}',tex)
        parsed=parser.parse('$'+tex.strip('$')+'$',dpi=72,prop=FontProperties(size=size*self.scale,math_fontfamily='stix'))
        import numpy as np
        alpha=Image.fromarray(np.asarray(parsed.image,dtype=np.uint8),'L')
        tile=Image.new('RGBA',alpha.size,color);tile.putalpha(alpha)
        bbox=tile.getbbox()
        baseline_offset=(parsed.height-parsed.depth-(bbox[1] if bbox else 0))/self.scale
        if bbox:tile=tile.crop(bbox)
        effective_size=size
        if self.strict_text and max_width and tile.width>max_width*self.scale:
            raise ValueError(f'Equation needs more space at {size}px: {tex!r}; '
                             f'width={tile.width/self.scale:.1f}, available={max_width}')
        if max_width and tile.width>max_width*self.scale:
            ratio=max_width*self.scale/tile.width
            tile=tile.resize((round(tile.width*ratio),max(1,round(tile.height*ratio))),Image.Resampling.LANCZOS)
            effective_size=size*ratio
            baseline_offset*=ratio
        self.formula_sizes.append((tex,size,effective_size))
        if align=='center':x-=tile.width/self.scale/2
        elif align=='right':x-=tile.width/self.scale
        elif align!='left':raise ValueError(f'Unknown equation alignment: {align}')
        if valign=='baseline':y-=baseline_offset
        elif valign!='top':raise ValueError(f'Unknown equation vertical alignment: {valign}')
        self.formula_layout.append({'tex':tex,'baseline':y+baseline_offset,
                                    'bounds':(x,y,x+tile.width/self.scale,y+tile.height/self.scale)})
        self.image.paste(tile,self._xy((x,y)),tile)
        self.text_bounds.append((x,y,x+tile.width/self.scale,y+tile.height/self.scale,'MATH '+tex))
        return tile.width/self.scale,tile.height/self.scale

    def tokens(self,x,y,n=5,fill='#b8d7c7',size=18):
        colors=fill if isinstance(fill,(list,tuple)) else [fill]*n
        for i in range(n):self.rect(x+i*(size+5),y,size,size*.7,colors[i%len(colors)],outline='#506477',r=3,width=1.4)

    def graph(self,x,y,w,h,labels=None,color='#467796',label_size=20,show_labels=True):
        names=list(labels or ['Drug','Enzyme','Organ','Parameter','Source'])
        n=len(names); coords=[(x+w*.50,y+h*.42)]
        angles=[math.pi*1.22,math.pi*1.77,math.pi*.15,math.pi*.64]
        for i in range(n-1):
            a=angles[i%len(angles)] if n<=5 else -math.pi*.8+2*math.pi*i/(n-1)
            coords.append((x+w*.50+math.cos(a)*w*.37,y+h*.47+math.sin(a)*h*.33))
        for i in range(1,n):self.line([coords[0],coords[i]],'#7e8d94',2,arrow=i%2==1)
        if n>3:self.line([coords[1],coords[-1]],'#adb7bd',1.4,dash=True)
        colors=['#8ebdb3','#adcbe1','#d7c2dc','#ecd59c','#f0bca7']
        rr=min(w,h)*.078
        for i,(xx,yy) in enumerate(coords):
            self.circle(xx,yy,rr,colors[i%5],color,1.7)
            if show_labels:
                self.text(xx,yy+rr+5,names[i],size=label_size,bold=i==0,align='center',max_width=w*.43)

    def icon(self,name,x,y,s=90,color='#467796'):
        k=s/100
        def r(a,b,w,h,fill,outline=color,rad=6):self.rect(x+a*k,y+b*k,w*k,h*k,fill,outline,r=rad*k,width=1.8*k)
        def e(a,b,w,h,fill,outline=color):self.ellipse(x+a*k,y+b*k,w*k,h*k,fill,outline,width=1.8*k)
        def l(points,c=color,wd=2,arrow=False):self.line([(x+a*k,y+b*k) for a,b in points],c,wd*k,arrow=arrow)
        def p(points,fill,outline=color):self.polygon([(x+a*k,y+b*k) for a,b in points],fill,outline,width=1.8*k)
        if name in ('paper','document'):
            r(17,21,57,68,'#e0eaf1');r(23,15,57,68,'#f8f5eb');p([(29,9),(70,9),(86,25),(86,78),(29,78)],'#ffffff');p([(70,9),(70,25),(86,25)],'#d8e8ed')
            for yy,ll in [(35,41),(45,45),(55,34),(65,39)]:l([(37,yy),(37+ll,yy)],'#8da9b6',1.5)
            e(7,67,24,24,'#deeedb');l([(15,78),(21,84),(30,70)],'#56876c',2.5)
        elif name in ('bottle','drug'):
            r(34,9,34,15,'#8ebbb6');r(28,24,45,62,'#fdf6e7',rad=10);r(31,41,39,31,'#e9eef5',outline='#c5d8e2',rad=1)
            l([(50,47),(50,64)],'#e19388',4);l([(42,56),(59,56)],'#e19388',4)
            p([(74,76),(87,70),(95,82),(81,91)],'#e6b0aa');l([(80,73),(88,86)],'#a8777e',1.4)
        elif name in ('pk','curve'):
            l([(13,14),(13,84),(93,84)],'#6c7b88',2)
            pts=[]
            for i in range(80):
                t=i/17;v=1.8*t*math.exp(-.8*t);pts.append((17+i*.88,79-62*v))
            l(pts,'#5c9890',3)
            for a,b in [(24,48),(35,28),(46,33),(64,48),(79,62)]:e(a-2,b-2,4,4,'#e5ad73',outline=None)
        elif name in ('scientist','researcher'):
            p([(20,93),(25,68),(40,60),(64,60),(80,72),(85,93)],'#e8eef3')
            p([(41,60),(50,79),(62,60),(63,88),(40,88)],'#b2d4c5');p([(36,61),(46,84),(35,77),(29,83),(31,67)],'#fff');p([(67,61),(57,84),(71,78),(77,84),(73,67)],'#fff')
            e(28,10,49,54,'#735e59');e(33,17,39,43,'#f4cfad');p([(31,31),(33,15),(57,13),(74,29),(59,21),(49,34),(43,27)],'#725f59')
            e(35,31,14,10,'#f8e6cd');e(56,31,14,10,'#f8e6cd');l([(49,35),(56,35)],wd=1.4);e(41,34,2.8,3.5,INK,None);e(61,34,2.8,3.5,INK,None)
            l([(47,49),(54,51),(60,48)],'#b87c6a',1.6);r(64,76,24,17,'#f5e8be',rad=2)
        elif name in ('robot','llm'):
            r(19,22,63,53,'#e4edf2',rad=13);e(7,38,13,25,'#a4b9d2');e(82,38,13,25,'#a4b9d2');r(28,33,45,27,'#c1dacf',rad=7)
            e(36,42,6,9,'#455d68',None);e(59,42,6,9,'#455d68',None);l([(42,57),(51,61),(60,57)],'#455d68',1.4)
            l([(51,21),(51,11)],wd=2);e(46,4,10,10,'#efc789');r(28,78,47,13,'#bccdde',rad=5)
            for angle in range(0,180,60):
                a=math.radians(angle);l([(88-11*math.cos(a),14-11*math.sin(a)),(88+11*math.cos(a),14+11*math.sin(a))],'#73b5cd',1.7)
        elif name in ('liver',):
            p([(9,38),(22,22),(51,18),(84,29),(95,40),(88,55),(72,64),(58,79),(43,73),(34,58),(16,60),(8,52)],'#cc9a87')
            l([(51,22),(45,40),(56,49),(64,65)],'#8e6660',2);l([(45,40),(28,42),(18,50)],'#a2766d',1.5);e(52,56,11,21,'#98b68a','#6e8c6d')
        elif name in ('kidney',):
            p([(32,15),(20,21),(12,40),(12,62),(23,80),(36,84),(44,74),(39,62),(29,55),(29,43),(43,34),(43,22)],'#c990aa')
            p([(69,15),(81,21),(89,40),(89,62),(78,80),(65,84),(57,74),(62,62),(72,55),(72,43),(58,34),(58,22)],'#c990aa')
            l([(40,46),(48,49),(46,94)],'#d3b077',2.8);l([(62,46),(53,49),(55,94)],'#d3b077',2.8)
            l([(41,37),(51,41),(60,36)],'#a36d82',2)
        elif name in ('gut','intestine'):
            r(17,16,69,69,'#f4ccaa',outline='#b78c77',rad=18)
            l([(30,78),(27,31),(73,31),(73,70),(40,70),(40,44),(60,44),(60,58),(51,58)],'#bd8c78',9)
            l([(30,78),(27,31),(73,31),(73,70),(40,70),(40,44),(60,44),(60,58),(51,58)],'#e5ad92',5)
        elif name in ('blood',):
            p([(51,8),(28,39),(20,60),(24,78),(38,89),(58,89),(77,76),(80,59),(66,34)],'#d39197')
            p([(50,23),(33,48),(30,60),(35,69),(43,59),(47,41)],'#f5c8c6',outline=None)
        elif name in ('encoder','network'):
            pts=[[(13,23),(13,51),(13,79)],[(49,14),(49,39),(49,65),(49,90)],[(85,30),(85,70)]]
            for a,b in zip(pts,pts[1:]):
                for u in a:
                    for v in b:l([u,v],'#b0bbc7',1)
            for i,layer in enumerate(pts):
                for a,b in layer:e(a-6,b-6,12,12,['#bfd4e1','#d5c4dd','#c0d8c9'][i])
        elif name in ('lock','gate'):
            e(24,8,50,59,'#fbfbfb');e(36,20,26,37,'white');r(18,42,64,47,'#e5d49a',rad=9);e(45,54,10,12,'#7a754f',None);r(47,62,6,12,'#7a754f',outline=None,rad=2)
        elif name in ('tissue',):
            for a,b,z in [(32,29,27),(64,32,27),(23,61,25),(55,65,28),(80,61,18)]:
                e(a-z/2,b-z/2,z,z,'#c5bada','#8c83ac');e(a-3,b-3,6,6,'#9385ae',None)
        else:
            self.graph(x,y,s,s,['','','','',''],color)

    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        bad=[b for b in self.text_bounds if b[0]<-.5 or b[1]<-.5 or b[2]>self.w+.5 or b[3]>self.h+.5]
        if bad:raise ValueError(f'Text exceeds canvas: {bad[:6]}')
        self.image.save(path,dpi=(300,300))
        return {'path':str(path),'size':self.image.size,'text_blocks':len(self.text_bounds),'bounds_ok':not bad,
                'text_sizes':sorted({actual for _,_,actual in self.text_sizes}),
                'auto_shrunk_labels':sum(requested!=actual for _,requested,actual in self.text_sizes),
                'formula_sizes':sorted({actual for _,_,actual in self.formula_sizes}),
                'auto_shrunk_equations':sum(requested!=actual for _,requested,actual in self.formula_sizes)}
