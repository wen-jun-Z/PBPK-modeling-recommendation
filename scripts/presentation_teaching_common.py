"""Shared, spacious layout for an introductory PBPK presentation."""
from build_blueprint_method_slides import *


def teaching_slide(title, subtitle, section):
    s = Slide(title, subtitle, section, '')
    s.ops[0]['rows'] = ['PBPK · First principles   /   ' + section]
    s.ops[4]['rows'] = ['PBPK · From inputs to predictions · Diagrams and hypothetical drug values are teaching examples, not new experiments']
    return s


def card(s, x, y, w, h, title, body='', color=TEAL, fill=PALE):
    s.rect(x, y, w, h, fill, color, .85)
    sizes = [(ts, bs) for ts in range(19, 14, -1) for bs in range(17, 13, -1)]
    chosen = next(((ts, bs) for ts, bs in sizes
                   if len(wrap(title, w-24, ts))*ts*1.15+3+14
                   + (len(wrap(body, w-24, bs))*bs*1.24+3 if body else 0)+5 <= h), None)
    assert chosen is not None, 'Card too short: '+title
    title_size, body_size = chosen
    title_h = s.text(x+12, y+10, w-24, title, title_size, color,
                     bold=True, align='center', line=1.15)
    if body:
        body_h = s.text(x+12, y+title_h+14, w-24, body, body_size, INK,
                        align='center', line=1.24)
        assert title_h+14+body_h+5 <= h, 'Card too short: '+title


def arrow(s, a, b, color=TEAL, dash=False, both=False):
    s.line([a, b], color, 1.8, dash=dash, arrow=True, both=both)


def words(s, blocks, say):
    """Two short explanations and an on-slide sentence the presenter can use."""
    assert len(blocks) == 2
    s.line([(590, 145), (590, 431)], 'DCE5E8', 1)
    for y, (title, body) in zip((153, 292), blocks):
        th = s.text(614, y, 302, title, fit_size(title,302,28,20,line=1.16), TEAL, bold=True, line=1.16)
        assert th <= 28, 'Shorten explanation heading: '+title
        bh = s.text(614, y+36, 302, body, fit_size(body,302,97,18,line=1.28), INK, line=1.28)
        assert bh <= 97, 'Shorten explanation body: '+body
    sayline(s, say)


def sayline(s, say):
    s.rect(42, 451, 876, 44, PALE, None)
    h = s.text(55, 456, 850, 'Key point: '+say, fit_size('Key point: '+say,850,40,16,line=1.15), TEAL, line=1.15)
    assert h <= 40, 'Shorten spoken summary: '+say


def example(s, text, y=371, h=60):
    s.rect(50, y, 522, h, GOLD_PALE, None)
    used = s.text(63, y+9, 496, text, fit_size(text,496,h-13,16), GOLD, line=1.2)
    assert used <= h-13, 'Shorten example: '+text


def eq(s, latex, y, h=52, x=52, w=520, color=INK):
    s.eq(x, y, w, h, latex, EQUATION_FONT_SIZE, color)


def label(s, text, x, y, w, color=MUTED, size=16):
    s.text(x, y, w, text, size, color, align='center', line=1.2)


def teaching_curve(s, x=88, y=185, w=425, h=170, *, dots=False, fill=False):
    curve(s, x, y, w, h, dots=dots, fill=fill, labels=False)
    s.text(x-22, y-34, w, 'Plasma concentration (drug per liter)', 16, TEAL)
    s.text(x+w-112, y+h+8, 130, 'Time after dose', 16, MUTED)
    if dots:
        s.ellipse(x+2, y+h+41, 8, fill=WHITE, stroke=GOLD, width=1.7)
        s.text(x+16, y+h+34, 210, 'Dots: hypothetical observations', 15, GOLD)
    return s


def make_foundation_slides():
    pages = []
    s = teaching_slide('After a dose, how much drug is in the body?',
                       'A beginner route: pharmacokinetics, blueprint construction, calibration, prediction, and validation.', 'Introduction 01')
    card(s, 57, 153, 504, 65, 'How do dose and infusion duration change concentration?')
    infusion(s, 83, 262, .9)
    s.asset('01_human_core.png', 242, 236, 92, 154)
    curve(s, 407, 267, 137, 88, labels=False)
    arrow(s, (143, 305), (223, 305))
    arrow(s, (339, 305), (395, 305))
    label(s, 'Input: drug dosing', 48, 395, 165, TEAL)
    label(s, 'Process: drug movement', 215, 395, 180, GREEN)
    label(s, 'Output: concentration vs time', 398, 395, 178, TEAL)
    words(s, [('What is PK?', 'Pharmacokinetics studies drug distribution and elimination, and how concentration changes over time.'),
              ('Our goal', 'When no suitable model exists, combine a human template, drug data, and a few observations to predict new dosing conditions.')],
          'We ask how concentration changes after dosing; this does not establish drug efficacy.')
    pages.append(s)

    s = teaching_slide('Learn to read the concentration curve',
                       'The curve is illustrative; dots are not newly collected clinical observations.', 'Introduction 02')
    teaching_curve(s, x=81, y=194, w=439, h=140, dots=True)
    label(s, 'Line: model prediction', 228, 383, 301, TEAL)
    words(s, [('Read the axes', 'Time after dosing is on the horizontal axis; plasma concentration is on the vertical axis. Plasma is the fluid part of blood.'),
              ('Lines and dots', 'The line is predicted; dots are blood measurements. Fit to one set of observations, then evaluate against another set.')],
          'Dots are measured, lines are calculated. Evaluate predictions using observations excluded from fitting.')
    pages.append(s)

    s = teaching_slide('Drug amount and concentration are different',
                       'Three units: mg for amount, L for volume, and h for time.', 'Introduction 03')
    for x, vol, height, color in [(91, '1 L', 76, TEAL), (339, '2 L', 111, GREEN)]:
        s.rect(x, 169, 144, 139, WHITE, color, 1.2, rounded=False)
        s.rect(x+2, 308-height, 140, height-2, PALE if color==TEAL else GREEN_PALE, None, rounded=False)
        for dx,dy in [(31,30),(70,30),(31,58),(70,58)]:
            s.ellipse(x+dx, 232+dy, 9, fill=GOLD, stroke=None)
        label(s, 'Same 4 mg dose', x-5, 144, 154, color)
        label(s, vol+' → '+('4' if vol=='1 L' else '2')+' mg/L', x-8, 317, 160, color)
    eq(s, r'C=\frac{A}{V}', 365, 54)
    words(s, [('Same amount, different density', '4 mg in 1 L gives 4 mg/L; the same amount in 2 L gives 2 mg/L.'),
              ('The simplest equation', 'A is amount, V is volume, and C is concentration. Tissue binding and partitioning later require an equivalent capacity.')],
          'Amount is total drug; concentration is drug per liter. PBPK calculates amounts, then converts to concentrations.')
    pages.append(s)

    s = teaching_slide('PBPK maps organs and blood flow',
                       'Physiologically based pharmacokinetics explicitly represents the physiological structure of the body.', 'Introduction 04')
    card(s, 72, 158, 477, 53, 'Circulation: drug travels to organs and returns')
    for x,title,body,col,fill in [(55,'Liver','One calculation unit',TEAL,PALE),
                                 (231,'Kidney','One calculation unit',GREEN,GREEN_PALE),
                                 (407,'Other tissue','Track drug amounts',GOLD,GOLD_PALE)]:
        card(s,x,279,151,102,title,body,col,fill)
        arrow(s,(x+75,215),(x+75,273),col,both=True)
    label(s,'Three regions are shown; the solver connects more organs.',56,401,504)
    words(s, [('Why use organ boxes?', 'Each box tracks drug amount. Arrows represent blood flow or drug exchange; the boxes are mathematical units.'),
              ('How is PBPK different?', 'A two-compartment model uses two effective spaces; PBPK uses physiology. More detail does not guarantee better predictions.')],
          'PBPK uses a physiological map to track drug amounts and calculate flows between regions.')
    pages.append(s)

    s = teaching_slide('Three stages, three different outputs',
                       'First separate the roles; then consider what happens when no suitable model exists.', 'Introduction 05')
    card(s,57,153,503,62,'① Components: find evidence-supported mechanisms')
    arrow(s,(308,219),(308,240))
    card(s,57,245,503,62,'② Models: find a suitable existing model',color=GREEN,fill=GREEN_PALE)
    arrow(s,(200,312),(170,348),GREEN)
    arrow(s,(419,312),(440,348),GOLD)
    card(s,52,355,221,67,'Yes: verify and reuse',color=GREEN,fill=GREEN_PALE)
    card(s,310,355,255,67,'No: ③ design a blueprint',color=GOLD,fill=GOLD_PALE)
    words(s, [('First retrieve and select', 'Component recommendation identifies mechanisms and evidence; model recommendation identifies suitable models. Neither produces a curve.'),
              ('Then build and calculate', 'If reuse is unavailable, combine drug data and physiology, estimate allowed parameters, generate curves, and evaluate predictions.')],
          'Find mechanisms, select models, then build a blueprint only when no suitable model can be reused.')
    pages.append(s)
    return pages


def make_teaching_conclusion_slides():
    pages = []
    s = teaching_slide('Current results: values and limits',
                       'These are existing retrospective results for 30 drugs, explained here without rerunning the experiments.', 'Interpreting results')
    card(s,55,150,507,59,'Same five development points → target-study comparison')
    columns=[(69,192,'Metric'),(275,124,'Blueprint'),(425,120,'Two-compartment')]
    for x,w,t in columns:
        s.text(x,237,w,t,fit_size(t,w,28,18,minimum=13),TEAL,bold=True,align='center')
    s.line([(61,271),(558,271)],LINE,1)
    for y,name,a,b in [(286,'Sampled peak GMFE','1.465','1.809'),(342,'Full-curve GMFE','1.553','1.703')]:
        s.rect(61,y-3,497,47,PALE,None,rounded=False)
        for x,w,t in [(69,192,name),(275,124,a),(425,120,b)]:
            s.text(x,y+7,w,t,18,TEAL if t==a else INK,align='center')
    label(s,'Equal points ≠ equal prior information; the comparator has no full physiology.',55,403,512,size=15)
    words(s,[('Reading these values', 'GMFE is better near 1. Average sampled-peak error is lower here; a sampled peak may differ from the continuous peak.'),
             ('Limits of interpretation', 'The paired curve interval includes no difference; some drugs worsen. These results do not establish universal gains or clinical safety.')],
          'Benefits apply under limited conditions; report drug scope, available information, and unresolved limitations.')
    pages.append(s)

    s=teaching_slide('The blueprint in four steps',
                     'End of the main presentation. Backup slides cover parameter mapping, differential equations, calibration, and endpoints.', 'One-minute recap')
    rows=[('01','Collect drug, human, dosing, and development data; check sources and units.',TEAL,PALE),
          ('02','Build equations from physiology and drug data; retain qualified measured parameters.',GREEN,GREEN_PALE),
          ('03','Fit unknowns to development observations; check whether bounded repair is needed.',TEAL,PALE),
          ('04','Lock the model, predict new conditions, and score against independent observations.',GOLD,GOLD_PALE)]
    for k,(num,text,col,fill) in enumerate(rows):
        y=147+k*72
        s.rect(50,y,864,60,fill,None)
        s.text(66,y+12,61,num,24,col,bold=True)
        h=s.text(138,y+14,752,text,fit_size(text,752,45,21,line=1.15),INK,line=1.15)
        assert h<46
    sayline(s,'Construction, permitted adjustments, and validation need evidence; a plausible-looking curve is insufficient.')
    pages.append(s)
    return pages
