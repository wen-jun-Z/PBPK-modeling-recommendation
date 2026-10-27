"""Render the four-lane PBPK methods preview with top-aligned explanations."""
from pathlib import Path
from math import hypot, floor, ceil, cos, sin, pi
from PIL import Image, ImageDraw
from english_workflow_art import Canvas

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/figures/workflow/02_four_lanes_en_preview.png'
ASSETS = OUT.parent / 'assets'
TITLE_BAND_EXTRA = 28
SIDEBAR_EXTRA = 80
BODY_EXTRA = {'a': 12, 'b': 22, 'c': 48, 'd': 20}


class LayoutCanvas(Canvas):
    """Translate whole content groups while keeping sidebar typography separate."""
    offset_x = 0

    def _at(self, method, x, *args, **kwargs):
        offset = self.offset_x
        self.offset_x = 0
        try:
            return method(x+offset, *args, **kwargs)
        finally:
            self.offset_x = offset

    def _points(self, method, points, *args, **kwargs):
        offset = self.offset_x
        self.offset_x = 0
        try:
            return method([(x+offset, y) for x,y in points], *args, **kwargs)
        finally:
            self.offset_x = offset

    def text(self, x, *args, **kwargs):
        return self._at(super().text, x, *args, **kwargs)

    def formula(self, x, *args, **kwargs):
        return self._at(super().formula, x, *args, **kwargs)

    def rect(self, x, *args, **kwargs):
        return self._at(super().rect, x, *args, **kwargs)

    def circle(self, x, *args, **kwargs):
        return self._at(super().circle, x, *args, **kwargs)

    def ellipse(self, x, *args, **kwargs):
        return self._at(super().ellipse, x, *args, **kwargs)

    def asset(self, path, x, *args, **kwargs):
        offset = self.offset_x
        self.offset_x = 0
        try:
            return super().asset(path, x+offset, *args, **kwargs)
        finally:
            self.offset_x = offset

    def tokens(self, x, *args, **kwargs):
        return self._at(super().tokens, x, *args, **kwargs)

    def graph(self, x, *args, **kwargs):
        return self._at(super().graph, x, *args, **kwargs)

    def icon(self, name, x, *args, **kwargs):
        offset = self.offset_x
        self.offset_x = 0
        try:
            return super().icon(name, x+offset, *args, **kwargs)
        finally:
            self.offset_x = offset

    def line(self, points, *args, **kwargs):
        return self._points(super().line, points, *args, **kwargs)

    def polygon(self, points, *args, **kwargs):
        return self._points(super().polygon, points, *args, **kwargs)


W, H = 2600+SIDEBAR_EXTRA, 1874+4*TITLE_BAND_EXTRA+sum(BODY_EXTRA.values())
c = LayoutCanvas(w=W, h=H, scale=2, strict_text=True)
INK, BLACK = '#293744', '#111111'
BLUE, GREEN, GOLD, PURPLE = '#557FA5', '#527D65', '#AD8545', '#8670A4'
A_FILL, B_FILL, C_FILL, D_FILL = '#EFF5FA', '#F0F6F1', '#FCF7E9', '#F3EFF8'
TEXT_ACCENTS = {BLUE: '#42688A', GREEN: '#426A53', GOLD: '#80602F', PURPLE: '#6B5489'}
CENTERED_LABELS = []
CONNECTORS = []
LANE_HEADERS = []
STAGE_HEADERS = []
SIDEBAR_TITLES = []
SYMBOL_MARKS = []
EQUATION_SIZE = 28


def txt(x, y, value, size=24, color=INK, bold=False, align='left', max_width=None):
    c.text(x, y, value, size=size, color=TEXT_ACCENTS.get(color, color), bold=bold, align=align, max_width=max_width)


def math(x, y, tex, max_width=None, align='center', baseline=False, size=22, color=BLACK):
    return c.formula(x, y, tex, size=size, max_width=max_width, color=color, align=align,
                     valign='baseline' if baseline else 'top')


def symbol(x, centre_y, tex, size=24, color=BLACK, max_width=80):
    """Label an illustrated object with the same variable as its equation."""
    measure = Canvas(w=1, h=1, scale=c.scale)
    _, height = measure.formula(0, 0, tex, size=size)
    math(x, centre_y-height/2, tex, size=size, color=color, max_width=max_width)
    SYMBOL_MARKS.append({'symbol': tex, 'bounds': c.text_bounds[-1]})



def object_label(x, centre_y, label, tex, size=22, math_size=24, gap=8):
    """Keep a symbol visibly attached to its named object, with aligned ink."""
    measure = Canvas(w=1, h=1, scale=c.scale)
    measure.text(0, 0, label, size=size)
    b = measure.text_bounds[-1]
    tw = b[2]-b[0]
    mw, _ = measure.formula(0, 0, tex, size=math_size)
    left = x-(tw+gap+mw)/2
    label_at(left, centre_y, label, size=size, color=BLACK, align='left')
    symbol(left+tw+gap+mw/2, centre_y, tex, size=math_size, max_width=mw+1)


def marked_organ_node(x, y, w, h, tex, color):
    """Mark the eligible Organ node, using the same layout as Canvas.graph."""
    cx = x+w*.50+cos(pi*1.77)*w*.37
    cy = y+h*.47+sin(pi*1.77)*h*.33
    c.circle(cx, cy, 15, fill='#D7C2DC', outline=color, width=1.7)
    symbol(cx, cy, tex, size=24, max_width=23)


def label_at(x, centre_y, value, size=22, color=INK, bold=False, align='center'):
    measure = Canvas(w=1, h=1, scale=c.scale)
    measure.text(0, 0, value, size=size, bold=bold)
    b = measure.text_bounds[-1]
    left = x-(b[2]-b[0])/2 if align=='center' else x
    txt(left-b[0], centre_y-(b[1]+b[3])/2, value, size=size, color=color, bold=bold)
    CENTERED_LABELS.append((centre_y, c.text_bounds[-1]))


def _rounded_path(points, radius):
    """Round only routing corners; preserve the exact source and target ports."""
    if not radius or len(points) < 3:
        return list(points)
    result = [points[0]]
    for a, b, d in zip(points, points[1:], points[2:]):
        la, ld = hypot(b[0]-a[0], b[1]-a[1]), hypot(d[0]-b[0], d[1]-b[1])
        if not la or not ld:
            continue
        r = min(radius, .4*la, .4*ld)
        p = (b[0]-(b[0]-a[0])*r/la, b[1]-(b[1]-a[1])*r/la)
        q = (b[0]+(d[0]-b[0])*r/ld, b[1]+(d[1]-b[1])*r/ld)
        result.append(p)
        for i in range(1, 13):
            t = i/12
            result.append(((1-t)**2*p[0]+2*(1-t)*t*b[0]+t*t*q[0],
                           (1-t)**2*p[1]+2*(1-t)*t*b[1]+t*t*q[1]))
    result.append(points[-1])
    return result


def _path_slice(points, start, end):
    result, distance = [], 0.
    for a, b in zip(points, points[1:]):
        length = hypot(b[0]-a[0], b[1]-a[1])
        if length and distance+length > start and distance < end:
            lo, hi = max(0., (start-distance)/length), min(1., (end-distance)/length)
            result.extend([(a[0]+(b[0]-a[0])*lo, a[1]+(b[1]-a[1])*lo),
                           (a[0]+(b[0]-a[0])*hi, a[1]+(b[1]-a[1])*hi)])
        distance += length
    return result


def connector(points, color=INK, width=2., dashed=False, radius=0, head=True, role='internal'):
    """Continuous dashes, rounded caps and tangent-aligned compact arrowheads."""
    points = _rounded_path(points, radius)
    length = sum(hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(points, points[1:]))
    if not length:
        return
    head_length = min(11., max(7., width*3+1), length*.43) if head else 0.
    shaft_end = length-head_length*.68

    def stroke(segment):
        if len(segment) < 2:
            return
        # Slicing adjoining segments repeats their shared vertices. Remove
        # raster duplicates before stroking; otherwise rounded joins can leave
        # little spurs that look like a solid/dashed overlay on steep curves.
        aa = 3
        pts = []
        for x, y in segment:
            point = (round((x+c.offset_x)*c.scale*aa), round(y*c.scale*aa))
            if not pts or point != pts[-1]:
                pts.append(point)
        if len(pts) < 2:
            return
        pad = ceil(width*c.scale/2)+3
        left = floor(min(p[0] for p in pts)/aa)-pad
        top = floor(min(p[1] for p in pts)/aa)-pad
        right = ceil(max(p[0] for p in pts)/aa)+pad
        bottom = ceil(max(p[1] for p in pts)/aa)+pad
        size = (right-left, bottom-top)
        mask = Image.new('L', (size[0]*aa, size[1]*aa), 0)
        draw = ImageDraw.Draw(mask)
        local = [(x-left*aa, y-top*aa) for x, y in pts]
        stroke_width = max(1, round(width*c.scale*aa))
        draw.line(local, fill=255, width=stroke_width, joint='curve')
        radius = stroke_width/2
        for x, y in (local[0], local[-1]):
            draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=255)
        mask = mask.resize(size, Image.Resampling.LANCZOS)
        c.image.paste(color, (left, top, right, bottom), mask)

    if dashed:
        distance = 0.
        # The dash phase follows the whole curve, never restarting per segment.
        while distance < shaft_end:
            stroke(_path_slice(points, distance, min(distance+8, shaft_end)))
            distance += 14
        if head:
            stroke(_path_slice(points, max(0, shaft_end-4), shaft_end))
    else:
        stroke(_path_slice(points, 0, shaft_end))
    polygon = []
    if head:
        a, tip = points[-2:]
        tangent = hypot(tip[0]-a[0], tip[1]-a[1])
        ux, uy = (tip[0]-a[0])/tangent, (tip[1]-a[1])/tangent
        wing = head_length*.40
        base = (tip[0]-ux*head_length, tip[1]-uy*head_length)
        polygon = [tip, (base[0]+uy*wing, base[1]-ux*wing),
                   (tip[0]-ux*head_length*.68, tip[1]-uy*head_length*.68),
                   (base[0]-uy*wing, base[1]+ux*wing)]
        c.polygon(polygon, fill=color)
    CONNECTORS.append({'points': points, 'width': width, 'head': polygon,
                       'dashed': dashed, 'role': role, 'offset_x': c.offset_x})


def curve(start, control1, control2, end, color=INK, width=2., dashed=False, head=True, role='internal'):
    points = []
    for i in range(49):
        t, u = i/48, 1-i/48
        points.append(tuple(u**3*start[k]+3*u*u*t*control1[k]+3*u*t*t*control2[k]+t**3*end[k]
                            for k in (0, 1)))
    connector(points, color, width, dashed=dashed, head=head, role=role)


def stacked_equalities(x, baselines, equations, size=EQUATION_SIZE):
    """Align a pair of full equations on the right edge of their equals signs."""
    probe = Canvas(w=1, h=1, scale=c.scale)
    prefixes = [probe.formula(0, 0, tex.split('=', 1)[0]+'=', size=size)[0]
                for tex in equations]
    widths = [probe.formula(0, 0, tex, size=size)[0] for tex in equations]
    left_extent = max(prefixes)
    right_extent = max(w-p for w,p in zip(widths,prefixes))
    equality_right = x+(left_extent-right_extent)/2
    for baseline, tex, prefix in zip(baselines, equations, prefixes):
        math(equality_right-prefix, baseline, tex, align='left', baseline=True, size=size)


def arrow(x1, y1, x2, y2, color=INK, dash=False, width=2):
    connector([(x1, y1), (x2, y2)], color, width, dashed=dash)


def flow(x1, y1, x2, y2, color=INK):
    connector([(x1, y1), (x2, y2)], color, width=3.2, role='main')


def lane(y, h, letter, title, fill, accent, principle):
    c.offset_x = 0
    c.rect(38, y, 2524+SIDEBAR_EXTRA, h, fill=fill, outline=accent, r=24, width=2)
    divider_x = 273+SIDEBAR_EXTRA
    c.line([(divider_x, y+24), (divider_x, y+h-24)], fill=accent, width=2)
    txt(63, y+26, f'({letter})', size=40, bold=True, color=accent)
    for i, label in enumerate(title.split('\n')):
        txt(63, y+94+i*46, label, size=36, bold=True, color=BLACK, max_width=268)
        SIDEBAR_TITLES.append({'letter': letter, 'bounds': c.text_bounds[-1],
                               'divider_x': divider_x, 'font_size': 36})
    c.offset_x = SIDEBAR_EXTRA
    # Give the main statement its own band; body geometry uses a separate
    # origin so extra breathing room never shrinks illustrations or equations.
    txt(1410, y+13, principle, size=32, bold=True, color=BLACK,
        align='center', max_width=2180)
    principle_bounds = c.text_bounds[-1]
    rule_y = y+64
    rule_color = {BLUE: '#D7E3EC', GREEN: '#D7E4DA',
                  GOLD: '#EBDDBF', PURPLE: '#E1D8ED'}[accent]
    connector([(298, rule_y), (2538, rule_y)], rule_color,
              width=1.1, head=False, role='header_rule')
    LANE_HEADERS.append({'letter': letter, 'top': y, 'height': h,
                         'body_y': y+TITLE_BAND_EXTRA, 'rule_y': rule_y,
                         'bounds': principle_bounds})


def step(x, y, heading, subtitle=None, width=None, color=INK):
    txt(x, y+55, heading, size=28, bold=True, color=color, align='center', max_width=width)
    STAGE_HEADERS.append({'body_y': y, 'bounds': c.text_bounds[-1]})
    if subtitle:
        txt(x, y+96, subtitle, size=24, bold=True, color=BLACK, align='center', max_width=width)


def neural(x, y, width=108, height=144, color=GREEN):
    positions = [[(x+li*width/2, y+(j+.5)*height/n) for j in range(n)]
                 for li, n in enumerate((3, 5, 3))]
    for a, b in zip(positions, positions[1:]):
        for p in a:
            for q in b:
                c.line([p, q], fill='#BCD0C4', width=1)
    for nodes in positions:
        for px, py in nodes:
            c.circle(px, py, 10, fill='#E5EFE8', outline=color, width=2)


def chip(x, y, w, label, fill, color=INK, h=38, size=22):
    c.rect(x, y, w, h, fill=fill, outline=color, r=10, width=1)
    label_at(x+w/2, y+h/2, label, size=size, color=color)


def small_doc(x, y, label, tag=None, scale=1):
    c.rect(x, y, 103*scale, 126*scale, fill='#FFFFFF', outline=BLUE, r=7, width=2)
    for i, width in enumerate((66, 73, 54, 68)):
        c.line([(x+15*scale, y+(34+16*i)*scale),
                (x+(15+width)*scale, y+(34+16*i)*scale)], fill='#A6B4BF', width=2)
    txt(x+51.5*scale, y+4, label, size=22, bold=True, align='center')
    if tag:
        chip(x+41*scale, y+91*scale, 66, tag, '#E9F2FA', BLUE)


def handoff_mark(x, y, scale=1):
    """Matching chain marks identify the real component output and graph input."""
    c.ellipse(x-17*scale, y-11*scale, 24*scale, 15*scale,
              fill=None, outline=GREEN, width=2.4*scale)
    c.ellipse(x-4*scale, y-2*scale, 24*scale, 15*scale,
              fill=None, outline=GOLD, width=2.4*scale)


txt(47, 27, 'Evidence-grounded PBPK recommendation and blueprint design',
    size=48, bold=True, color=BLACK)

# (a) Illustrations occupy the middle band; encoding steps stay below it.
A_TOP, HA = 108, 354+TITLE_BAND_EXTRA+BODY_EXTRA['a']
YA = A_TOP+TITLE_BAND_EXTRA
lane(A_TOP, HA, 'a', 'Knowledge\nconstruction', A_FILL, BLUE,
     'Preserve source spans, normalized units and model versions.')
step(450, YA, 'Evidence & model sources')
txt(450, YA+97, 'Literature · labels · PK studies', size=22, align='center')
c.icon('paper', 293, YA+136, s=146, color=BLUE)
c.icon('bottle', 404, YA+146, s=124, color=BLUE)
c.icon('pk', 513, YA+148, s=122, color=BLUE)
flow(644, YA+205, 681, YA+205, BLUE)

step(835, YA, 'Rule-first extraction', 'LLM-assisted review', width=346)
c.icon('robot', 697, YA+136, s=153, color=BLUE)
small_doc(869, YA+139, 'Facts', tag='[E1]', scale=1.25)
arrow(845, YA+205, 860, YA+205, BLUE)
flow(1020, YA+205, 1060, YA+205, BLUE)

step(1280, YA, 'Heterogeneous evidence graph', width=440)
nodes = {'Drug': (1115, YA+169), 'Enzyme': (1350, YA+168),
         'Organ': (1458, YA+272), 'Parameter': (1148, YA+295)}
curve((1169, YA+169), (1209, YA+181), (1256, YA+179), (1296, YA+168), BLUE, width=1.9)
curve((1389, YA+206), (1401, YA+215), (1406, YA+226), (1420, YA+235), GREEN, width=1.9)
curve((1129, YA+222), (1128, YA+229), (1130, YA+235), (1134, YA+240), GOLD, width=1.9)
txt(1233, YA+133, 'metabolized by', size=22, align='center')
txt(1350, YA+237, 'expressed in', size=22, align='center')
txt(1167, YA+213, 'has parameter', size=22)
for label, (x, y) in nodes.items():
    fill = {'Drug': '#F5D5C7', 'Enzyme': '#C6DED1',
            'Organ': '#DBD4EA', 'Parameter': '#F4E2B5'}[label]
    c.circle(x, y, 54 if label == 'Parameter' else 50,
             fill=fill, outline=BLUE, width=2)
    label_at(x, y, label, size=24, bold=True)
chip(1245, YA+299, 74, '[E1]', '#FFFFFF', BLUE)
chip(1335, YA+299, 74, '[E2]', '#FFFFFF', BLUE)
object_label(1260, YA+101, 'Subgraph', r'G_i')
flow(1528, YA+205, 1584, YA+205, BLUE)

step(1735, YA, 'Semantic index', 'Frozen text encoder', width=368)
c.icon('encoder', 1605, YA+132, s=138, color=BLUE)
c.tokens(1790, YA+197, n=5, fill='#BBD0E4', size=27)
symbol(1867, YA+177, r'\hat e_i', size=24)
flow(1748, YA+206, 1779, YA+206, BLUE)
object_label(1675, YA+292, 'Encoder', r'f')
enc_box = math(1650, YA+340, r'e_i=f(\mathrm{text}(G_i))', max_width=190, baseline=True, size=24)
norm_box = math(1860, YA+340, r'\hat e_i=e_i/\Vert e_i\Vert_2', max_width=190, baseline=True, size=24)
arrow(1650+enc_box[0]/2+9, YA+334, 1860-norm_box[0]/2-9, YA+334, BLUE)
step(2245, YA, 'Versioned model catalog', 'Structures · mechanisms · scope', width=500)
for j in range(3):
    small_doc(2010+166*j, YA+143, f'Model {j+1}', scale=1.4)

# (b) All method captions are above visuals; entity names and equations stay
# with their illustrated mechanisms.
B_TOP, HB = A_TOP+HA+16, 422+TITLE_BAND_EXTRA+BODY_EXTRA['b']
YB = B_TOP+TITLE_BAND_EXTRA
lane(B_TOP, HB, 'b', 'Component\nrecommendation', B_FILL, GREEN,
     'Train attention pooling and the shared projector; keep the LLM frozen.')
step(458, YB, 'Modeling request', width=336)
object_label(458, YB+103, 'Query', r'q')
c.rect(299, YB+119, 326, 190, fill='#FFFFFF', outline=GREEN, r=22, width=2)
for i, label in enumerate(('Drug + population', 'Route + dose', 'Prediction goal')):
    txt(462, YB+139+i*35, label, size=26, align='center')
math(462, YB+282, r'\hat e_q=\operatorname{norm}(f(q))', max_width=286, baseline=True, size=EQUATION_SIZE)
c.polygon([(441, YB+309), (437, YB+327), (464, YB+309)],
          fill='#FFFFFF', outline=GREEN, width=2)
c.icon('scientist', 398, YB+313, s=88, color=GREEN)
connector([(633, YB+216), (660, YB+216)], GREEN, width=3.2, head=False, role='main')

step(793, YB, 'Hybrid retrieval', 'Dense similarity + RRF', width=320)
chip(683, YB+145, 206, 'Lexical search', '#FFFFFF', GREEN, h=42)
c.rect(683, YB+227, 206, 85, fill='#FFFFFF', outline=BLUE, r=10, width=1)
label_at(786, YB+246, 'Dense search', size=22, color=BLUE)
math(786, YB+294, r's_{\mathrm{dense}}=\hat e_q^{T}\hat e_i', max_width=186, baseline=True, size=EQUATION_SIZE)
connector([(660, YB+216), (660, YB+166), (677, YB+166)], GREEN, width=2.3, radius=9)
connector([(660, YB+216), (660, YB+269), (677, YB+269)], BLUE, width=2.3, radius=9)
connector([(891, YB+166), (936, YB+166), (936, YB+193)], GREEN, width=2.3, radius=12, head=False)
connector([(891, YB+269), (936, YB+269), (936, YB+239)], BLUE, width=2.3, radius=10, head=False)
c.circle(936, YB+216, 23, fill='#D9E9DD', outline=GREEN, width=2)
label_at(936, YB+216, 'RRF', size=20, bold=True, color=BLACK)
math(793, YB+386, r's_{\mathrm{RRF}}(G_i)=\sum_r\frac{1}{60+\operatorname{rank}_r(G_i)}', max_width=316, baseline=True, size=EQUATION_SIZE)
flow(964, YB+216, 1000, YB+216, GREEN)

step(1135, YB, 'Graph-aware reranking', 'Cross-encoder + HGT', width=284)
c.graph(999, YB+135, 278, 188, labels=['Drug', 'Enzyme', 'Organ', 'PK', 'Study'],
        color=GREEN, label_size=22)
object_label(1135, YB+337, 'Retrieved', r'G_i')
math(1135, YB+378, r'h_i=\operatorname{Enc}_{\mathrm{HGT}}(q,G_i)', max_width=250, baseline=True, size=EQUATION_SIZE)
flow(1280, YB+216, 1314, YB+216, GREEN)
symbol(1297, YB+191, r'h_i')

step(1410, YB, 'Attention pooling', 'Weight, then sum', width=242)
for yy, rr in ((164, 15), (216, 21), (268, 17)):
    c.circle(1340, YB+yy, rr, fill='#ACD0BA', outline=GREEN, width=2)
    curve((1365, YB+yy), (1373, YB+yy), (1373, YB+216), (1380, YB+216),
          '#90AF9B', width=1.8, head=False)
c.rect(1380, YB+178, 139, 80, fill='#DDEBDD', outline=GREEN, r=22, width=2)
symbol(1449.5, YB+218, r'h_0=\sum_i a_i h_i', size=28, max_width=121)
math(1410, YB+324, r'a_i=\operatorname{softmax}_i(w^{T}h_i)', max_width=255, baseline=True, size=EQUATION_SIZE)
flow(1525, YB+216, 1537, YB+216, GREEN)

step(1673, YB, 'Token projection', 'Shared MLP projector', width=255)
neural(1552, YB+149)
# MLP produces green graph tokens; cited text remains a separate input.
connector([(1673, YB+216), (1686, YB+216), (1686, YB+177), (1696, YB+177)],
          GREEN, width=2.4, radius=6)
c.tokens(1706, YB+170, n=4, fill='#ACD0BA', size=20)
math(1750, YB+156, r'T=[t_0,t_1,\ldots,t_n]', max_width=187, baseline=True, size=22)
txt(1750, YB+225, 'Cited text', size=22, align='center')
c.tokens(1706, YB+255, n=4, fill='#EFC9B5', size=20)
# Individual subgraphs and the pooled state share the same projector.
connector([(1281, YB+216), (1281, YB+136),
           (1606, YB+136), (1606, YB+150)], GREEN, width=1.8,
          dashed=True, radius=6, role='conditioning')
c.circle(1281, YB+216, 2.6, fill=GREEN)
symbol(1518, YB+157, r'h_i')
math(1624, YB+324, r't_i=P_{\phi}(h_i)', max_width=220, baseline=True, size=EQUATION_SIZE)
math(1624, YB+371, r't_0=P_{\phi}(h_0)', max_width=250, baseline=True, size=EQUATION_SIZE)
# Both curved inputs meet at the same port; evidence conditioning is dashed.
curve((1807, YB+177), (1840, YB+177), (1808, YB+216), (1832, YB+216),
      GREEN, width=2.5, head=False)
curve((1807, YB+262), (1840, YB+262), (1808, YB+216), (1832, YB+216),
      '#BA8169', width=1.9, dashed=True, head=False, role='conditioning')
c.circle(1832, YB+216, 2.6, fill=GREEN)
arrow(1834, YB+216, 1850, YB+216, GREEN, width=2.5)

step(1940, YB, 'Grounded generation', 'Frozen LLM', width=272)
c.icon('robot', 1868, YB+136, s=162, color=GREEN)
c.icon('lock', 2024, YB+141, s=60, color=BLUE)
math(1940, YB+335, r'\operatorname{LLM}(q,T,\mathrm{text})', max_width=252, baseline=True, size=EQUATION_SIZE)
flow(2055, YB+216, 2102, YB+216, GREEN)

step(2332, YB, 'Evidence-backed components', 'Citation & mechanism checks', width=428)
for xx, icon, label in [(2110, 'gut', 'Absorption'), (2230, 'blood', 'Distribution'),
                        (2350, 'liver', 'Metabolism'), (2470, 'kidney', 'Excretion')]:
    c.icon(icon, xx, YB+161, s=88, color=GREEN)
    txt(xx+44, YB+252, label, size=22, align='center')
chip(2230, YB+290, 81, '[E1]', '#FFFFFF', GREEN)
chip(2320, YB+290, 81, '[E2]', '#FFFFFF', GREEN)
arrow(2332, YB+335, 2332, YB+352, GREEN)
c.rect(2115, YB+360, 393, 60, fill='#E1EDE3', outline=GREEN, r=13, width=1.5)
handoff_mark(2151, YB+390, .8)
label_at(2340, YB+390, 'Component specification', size=26, bold=True, color=BLACK)

# (c) Applicability tiers and within-tier relevance remain distinct decisions.
C_TOP, HC = B_TOP+HB+88, 394+TITLE_BAND_EXTRA+BODY_EXTRA['c']
YC = C_TOP+TITLE_BAND_EXTRA
lane(C_TOP, HC, 'c', 'Model\nrecommendation', C_FILL, GOLD,
     'Applicability first: rank model candidates within each qualified tier.')
step(414, YC, 'Requirement graph', color=GREEN)
step(707, YC, 'Model-version graphs')
txt(554, YC+96, 'Match eligible nodes within type', size=24, bold=True,
    color=BLACK, align='center', max_width=572)
c.graph(290, YC+140, 248, 182, labels=['Drug', 'Route', 'Organ', 'ADME'], color=GOLD, label_size=22)
c.graph(582, YC+140, 253, 182, labels=['Drug', 'Route', 'Organ', 'ADME'], color=GOLD, label_size=22)
marked_organ_node(290, YC+140, 248, 182, r'v', GOLD)
marked_organ_node(582, YC+140, 253, 182, r'u', GOLD)
object_label(414, YC+330, 'Graph', r'G_q')
object_label(707, YC+330, 'Graph', r'G_M')
handoff_mark(414, YC+137, .75)
flow(414, YC+155, 414, YC+195, GREEN)
connector([(535, YC+179), (577, YC+179)], GOLD, width=1.6, dashed=True, head=False, role='alignment')
connector([(535, YC+259), (577, YC+259)], GREEN, width=1.6, dashed=True, head=False, role='alignment')
flow(852, YC+235, 875, YC+235, GOLD)

step(1010, YC, 'Typed alignment', width=244)
txt(1010, YC+92, 'Same-type pairs', size=24, bold=True, color=BLACK, align='center')
matrix_colors = ['#F4E9CE', '#D8C18E', '#B79050', '#EEE2C5']
node_types = [0, 1, 1, 2, 2]
for i in range(5):
    for j in range(5):
        compatible = node_types[i] == node_types[j]
        co = ('#9B733A' if i == j else matrix_colors[(i*3+j) % 4]) if compatible else '#EBE7DD'
        c.rect(905+j*43, YC+143+i*43, 39, 39, fill=co, r=2)
        if not compatible:
            c.line([(916+j*43, YC+163+i*43), (933+j*43, YC+163+i*43)], '#C1BAAA', width=1.3)
symbol(892, YC+248.5, r'v')
symbol(1010.5, YC+131, r'u')
symbol(1010.5, YC+248.5, r's_{vu}', size=22, color='#FFFFFF', max_width=36)
u_box = math(425, YC+400, r'U_v=\{u:\mathrm{type}(u)=\mathrm{type}(v)\}', max_width=273, baseline=True)
m_box = math(720, YC+400, r'm_v=\max(\{s_{vu}:u\in U_v\}\cup\{0\})', max_width=280, baseline=True)
s_box = math(1010, YC+400, r'S_{\mathrm{type}}=\frac{\sum_v w_vm_v}{\sum_v w_v}', max_width=246, baseline=True, size=26)
arrow(425+u_box[0]/2+10, YC+394, 720-m_box[0]/2-10, YC+394, GOLD)
arrow(720+m_box[0]/2+12, YC+394, 1010-s_box[0]/2-12, YC+394, GOLD)
flow(1127, YC+235, 1160, YC+235, GOLD)

step(1280, YC, 'Compatibility gates', '1. Applicability first', width=278)
for k, label in enumerate(('Mechanism', 'Population', 'Route')):
    chip(1173, YC+146+67*k, 214, label, '#FFFFFF', GOLD, h=48, size=24)
flow(1399, YC+235, 1460, YC+235, GOLD)
step(1615, YC, 'Applicability tiers', '2. Relevance within tier', width=314)
tiers = [('1', 'Reuse', '#DDECDC'), ('2', 'Adapt', '#E4ECDD'),
         ('3', 'Reference', '#EEE5CE'), ('4', 'Reject', '#E9E0EF')]
for k, (num, label, fill) in enumerate(tiers):
    c.rect(1475, YC+145+53*k, 266, 45, fill=fill, outline=GOLD, r=10, width=1)
    label_at(1499, YC+167.5+53*k, num, size=24, bold=True, color=GOLD)
    label_at(1533, YC+167.5+53*k, label, size=26, align='left')
# Give every decision the same text column, type size and ink-centred port.
# The decision hub follows the actual vertical centre of the four tier rows.
DECISION_FRAMES = []


def decision_label(x, centre_y, lines, size=26, gap=8, framed=False):
    measure = Canvas(w=1, h=1, scale=c.scale)
    metrics = []
    for value, bold in lines:
        measure.text(0, 0, value, size=size, bold=bold)
        metrics.append(measure.text_bounds[-1])
    height = sum(b[3]-b[1] for b in metrics)+gap*(len(lines)-1)
    top = centre_y-height/2
    if framed:
        width = max(b[2]-b[0] for b in metrics)
        frame = (x-14, top-11, x+width+14, top+height+11)
        c.rect(frame[0], frame[1], frame[2]-frame[0], frame[3]-frame[1],
               fill='#EAF2EC', outline='#8DAC96', r=14, width=1.3)
        DECISION_FRAMES.append(frame)
    begin = len(c.text_bounds)
    for (value, bold), b in zip(lines, metrics):
        txt(x-b[0], top-b[1], value, size=size, bold=bold, color=BLACK)
        top += b[3]-b[1]+gap
    return c.text_bounds[begin:]


# First tier starts at +145; final tier ends at +349: centre = +247.
decision_mid = YC+247
reuse_y, reference_y, blueprint_y = decision_mid-80, decision_mid, decision_mid+80
connector([(1755, decision_mid), (1800, decision_mid)],
          GOLD, width=3.2, head=False, role='main')
curve((1800, decision_mid), (1800, reuse_y), (1820, reuse_y), (1852, reuse_y),
      GREEN, width=2.8, dashed=False, role='decision')
connector([(1800, decision_mid), (1857, reference_y)],
          INK, width=2., dashed=False, role='decision')
curve((1800, decision_mid), (1800, blueprint_y), (1820, blueprint_y), (1857, blueprint_y),
      PURPLE, width=2.2, dashed=False, role='decision')
c.circle(1800, decision_mid, 3.2, fill=GOLD)
DECISION_LABELS = [
    decision_label(1872, reuse_y, [('Reuse / controlled', True), ('adaptation', True)], framed=True),
    decision_label(1872, reference_y, [('Reference / abstain', True)]),
    decision_label(1872, blueprint_y, [('No reliable reusable model', True),
                                    ('Proceed to blueprint design', False)]),
]
REUSE_FRAME = DECISION_FRAMES[0]

# A prediction report distinguishes reuse from the blueprint artifact in (d).
# Integrate application into the output panel rather than showing a UI button.
# A fine curved link lands at the concentration plot's vertical centre.
report_x, report_y, report_size = 2240, YC+99, 284
report_port = (report_x+report_size*40/1254, report_y+report_size*(216+876)/(2*1254))
step(2382, YC, 'Model prediction', width=348)
c.asset(ASSETS/'model_prediction_output.png', report_x, report_y, report_size, report_size)
APPLICATION_START = (REUSE_FRAME[2], reuse_y)
APPLICATION_COLOR = '#638C77'
# Short horizontal entries and a symmetric eased bend give the line clean
# tangents at both ends. The open arrowhead stays separate from the panel.
application_tip = (report_port[0]-4, report_port[1])
bend_start = (APPLICATION_START[0]+22, reuse_y)
bend_end = (application_tip[0]-20, application_tip[1])
control_a = (bend_start[0]+46, bend_start[1])
control_b = (bend_end[0]-46, bend_end[1])
application_points = [APPLICATION_START, bend_start]
for i in range(1, 65):
    t, u = i/64, 1-i/64
    application_points.append(tuple(u**3*bend_start[k]+3*u*u*t*control_a[k]
                                    +3*u*t*t*control_b[k]+t**3*bend_end[k]
                                    for k in (0, 1)))
application_points.append(application_tip)
connector(application_points, APPLICATION_COLOR, width=1.9,
          head=False, role='application_link')
# Keep Run as an action label above the link, with real ink-to-line clearance.
# A sloping curve must be checked across the entire word, not only its centre.
application_path = CONNECTORS[-1]['points']
run_cx = (APPLICATION_START[0]+report_port[0])/2
run_measure = Canvas(w=1, h=1, scale=c.scale)
run_measure.text(0, 0, 'Run', size=24, bold=True)
run_metric = run_measure.text_bounds[-1]
run_width = run_metric[2]-run_metric[0]
run_left, run_right = run_cx-run_width/2, run_cx+run_width/2
run_line_top = min(y for x,y in application_path if run_left-2 <= x <= run_right+2)
RUN_LABEL_GAP = 12
run_top = run_line_top-RUN_LABEL_GAP-(run_metric[3]-run_metric[1])
txt(run_left-run_metric[0], run_top-run_metric[1], 'Run', size=24, bold=True, color=GREEN)
RUN_LABEL_BOUNDS = c.text_bounds[-1]
connector([(application_tip[0]-7, application_tip[1]-3.5), application_tip,
           (application_tip[0]-7, application_tip[1]+3.5)],
          APPLICATION_COLOR, width=1.5, head=False, role='application_open_head')
txt(2273, YC+146, 'Concentration', size=22, color=BLACK)
txt(2455, YC+252, 'Time', size=22, color=BLACK)
for centre_y, label, color, dashed in [(YC+194, 'Dose A', GREEN, False),
                                      (YC+224, 'Dose B', BLUE, True)]:
    connector([(2407, centre_y), (2430, centre_y)], color, width=2.2,
              dashed=dashed, head=False, role='profile_legend')
    decision_label(2440, centre_y, [(label, False)], size=22)
txt(2325, YC+306, 'AUC', size=22, bold=True, color=BLACK)
txt(2455, YC+306, 'Cmax', size=22, bold=True, color=BLACK)
txt(2382, YC+370, 'Illustrative', size=18, color=INK, align='center')

# (d) Short blood/organ connectors retain the original physiology. Equations
# follow the dependency: parameter -> flux -> balance -> amount -> concentration.
D_TOP, HD = C_TOP+HC+16, 450+TITLE_BAND_EXTRA+BODY_EXTRA['d']
YD = D_TOP+TITLE_BAND_EXTRA
lane(D_TOP, HD, 'd', 'PBPK blueprint\ndesign', D_FILL, PURPLE,
     'Bind evidence, sum process fluxes, conserve mass, solve and validate concentration profiles.')
step(489, YD, 'Bind sourced parameters', width=406)
small_doc(309, YD+141, 'Template', scale=1.1)
small_doc(437, YD+141, 'Evidence', scale=1.1)
c.rect(482.1, YD+241.1, 66, 38, fill='#E9F2FA', outline=BLUE, r=10, width=1)
symbol(515.1, YD+260.1, r'E_j')
c.rect(568, YD+151, 138, 148, fill='#FFFFFF', outline=PURPLE, r=12, width=2)
object_label(637, YD+174, 'Parameters', r'\theta', size=20, math_size=24)
for i, label in enumerate(('Value / unit', 'Context', 'Source')):
    txt(637, YD+198+i*30, label, size=22, align='center')
connector([(366, YD+284), (366, YD+315), (555, YD+315),
           (555, YD+217), (564, YD+217)], PURPLE, width=2.1, radius=9)
connector([(494, YD+284), (494, YD+315)], PURPLE, width=2.1, head=False)
math(489, YD+351, r'\theta_j\leftarrow\operatorname{value}(E_j)', max_width=340, baseline=True, size=EQUATION_SIZE)
arrow(716, YD+216, 730, YD+216, PURPLE)

step(916, YD, 'Assemble process fluxes', width=374)
flux_y = YD+70
arrow(889, flux_y+155, 830, flux_y+105, PURPLE)   # blood -> gut
arrow(953, flux_y+155, 1011, flux_y+101, PURPLE) # blood -> kidney
arrow(1010, flux_y+129, 954, flux_y+179, PURPLE) # kidney -> blood
arrow(889, flux_y+184, 832, flux_y+229, PURPLE)  # arterial supply to liver
arrow(832, flux_y+262, 889, flux_y+213, PURPLE)  # liver -> blood
arrow(953, flux_y+189, 1011, flux_y+229, PURPLE) # blood -> tissues
arrow(1011, flux_y+263, 953, flux_y+211, PURPLE) # tissues -> blood
connector([(762, flux_y+112), (741, flux_y+112), (741, flux_y+239), (754, flux_y+239)],
          PURPLE, width=1.9, radius=7, role='physiology')
for icon, x, y, label in [
    ('gut', 756, flux_y+59, 'Gut'), ('kidney', 1018, flux_y+59, 'Kidney'),
    ('blood', 884, flux_y+137, 'Blood'),
    ('liver', 756, flux_y+209, 'Liver'), ('tissue', 1018, flux_y+209, 'Tissues')]:
    c.icon(icon, x-10, y-10, s=90, color=PURPLE)
    if label in ('Blood', 'Kidney'):
        object_label(x+35, y+92, label, 'i' if label == 'Blood' else 'j')
    else:
        txt(x+35, y+78, label, size=22, align='center')
symbol(1070, YD+252, r'J_{j\rightarrow i}', max_width=90)
connector([(1031, YD+252), (984, YD+222)], PURPLE, width=1.1, head=False, role='callout')
txt(772, flux_y+173, 'Portal', size=22)
stacked_equalities(916, (YD+361, YD+425), [r'J_{\mathrm{in},i}=\sum_j J_{j\rightarrow i}',
                                    r'J_{\mathrm{out},i}=\sum_j J_{i\rightarrow j}'])
flow(1116, YD+216, 1162, YD+216, PURPLE)

step(1305, YD, 'Check the balance', 'Units · sources · mass', width=326)
c.icon('lock', 1271, YD+134, s=68, color=PURPLE)
balance_first = r'\frac{dA_i}{dt}=J_{\mathrm{in},i}-J_{\mathrm{out},i}'
balance_second = r'+R_{\mathrm{in},i}-R_{\mathrm{elim},i}'
formula_probe = Canvas(w=1, h=1, scale=c.scale)
balance_indent = formula_probe.formula(0, 0, r'\frac{dA_i}{dt}=', size=EQUATION_SIZE)[0]
balance_width = max(formula_probe.formula(0, 0, balance_first, size=EQUATION_SIZE)[0],
                    balance_indent+formula_probe.formula(0, 0, balance_second, size=EQUATION_SIZE)[0])
balance_left = 1305-balance_width/2
math(balance_left, YD+246, balance_first, max_width=278, align='left', baseline=True, size=EQUATION_SIZE)
math(balance_left+balance_indent, YD+291, balance_second,
     max_width=230, align='left', baseline=True, size=EQUATION_SIZE)
flow(1481, YD+216, 1495, YD+216, PURPLE)

step(1625, YD, 'Solve the ODEs', 'Versioned model files', width=255)
c.rect(1503, YD+151, 244, 178, fill='#FFFFFF', outline=PURPLE, r=14, width=2)
math(1625, YD+205, r'\operatorname{SolveODE}(\theta,A(0))', max_width=218, baseline=True, size=EQUATION_SIZE)
arrow(1625, YD+225, 1625, YD+245, PURPLE, width=1.7)
symbol(1625, YD+282, r'A(t)', size=30, max_width=150)
flow(1755, YD+216, 1810, YD+216, PURPLE)
symbol(1782.5, YD+190, r'A_i(t)', size=22, max_width=70)

step(1894, YD, 'Compute concentration', width=300)
object_label(1894, YD+110, 'Volume', r'V_i')
symbol(1853, YD+155, r'C_i(t)', size=24)
c.icon('pk', 1798, YD+148, s=192, color=PURPLE)
object_label(1945, YD+331, 'Time', r't')
math(1894, YD+386, r'C_i(t)=\frac{A_i(t)}{V_i}', max_width=236, baseline=True, size=EQUATION_SIZE)
flow(2013, YD+216, 2062, YD+216, PURPLE)

step(2303, YD, 'Blueprint + validation report', 'Executable model + assessment', width=454)
c.asset(ASSETS/'blueprint_validation.png', 2078, YD+135, 451, 314)

# Compact centered bridge touches the lane borders, not intermediate modules.
# Matching chain marks locate the real source and sink without a perimeter line.
bridge_x, bridge_y, bridge_w, bridge_h = 925, B_TOP+HB+17, 970, 56
bridge_mid = bridge_x+bridge_w/2
connector([(bridge_mid, B_TOP+HB), (bridge_mid, bridge_y)], GREEN, width=4.2, head=False, role='handoff')
c.rect(bridge_x, bridge_y, bridge_w, bridge_h, fill='#E3EEE5', r=20)
c.rect(bridge_mid, bridge_y+1, bridge_w/2-1, bridge_h-2, fill='#F5E9CD', r=19)
c.rect(bridge_mid, bridge_y+1, 25, bridge_h-2, fill='#F5E9CD', r=0)
c.rect(bridge_x, bridge_y, bridge_w, bridge_h, fill=None, outline=GREEN, r=20, width=2)
c.circle(bridge_mid, bridge_y+bridge_h/2, 22, fill='#FFFFFF', outline=GREEN, width=1.5)
handoff_mark(bridge_mid, bridge_y+bridge_h/2, .8)
label_at(1160, bridge_y+bridge_h/2, 'Components + cited evidence', size=24, bold=True, color=GREEN)
label_at(1650, bridge_y+bridge_h/2, 'Requirement graph', size=24, bold=True, color=GOLD)
arrow(bridge_mid, bridge_y+bridge_h+2, bridge_mid, C_TOP-3, GOLD, width=3)

# Narrow-gap arrows stay centred in the widened title rail.
c.offset_x = 0
rail_cx = (38+273+SIDEBAR_EXTRA)/2
arrow(rail_cx, A_TOP+HA+2, rail_cx, B_TOP-2, BLUE)
arrow(rail_cx, C_TOP+HC+2, rail_cx, D_TOP-2, PURPLE)
print(c.save(OUT))
