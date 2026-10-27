"""English mechanism preview, locally drawn after image service failures.

The four-panel composition follows the user's first layout. Scientific content
is grounded in online_retrieval, soft_graph_prompt, model_matching/scoring and
the blueprint templates. This is a framework illustration, not result data.
"""
from pathlib import Path
from english_workflow_art import Canvas

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/figures/workflow/01_four_panel_en_preview.png'
INK, GRAY, KEY = '#293744', '#53616C', '#111111'
BLUE, GREEN, GOLD, PURPLE = '#4F7DA4', '#58846E', '#AD8741', '#8D70A5'
COLORS = [BLUE, GREEN, GOLD, PURPLE]
MATH = 22


def arrow(c, p, col=GRAY, width=2.5, dash=False):
    c.line(p,fill=col,width=width,arrow=True,dash=dash)


def typed_graph(c,x,y,w,h,model=False):
    positions=[(.10,.5),(.52,.08),(.9,.48),(.53,.91)]
    names=['Drug','Route','Population','Mechanism']
    points=[(x+xx*w,y+yy*h) for xx,yy in positions]
    for a,b in ((0,1),(0,2),(0,3),(1,2),(2,3)):
        c.line([points[a],points[b]],fill='#AAB4BA',width=2)
    for i,((xx,yy),name) in enumerate(zip(points,names)):
        c.circle(xx,yy,16,fill=['#ABC7E1','#C4D8BD','#E9CBA5','#D5C6E0'][i],outline=COLORS[i],width=2)
        c.text(xx,yy+23,name,size=20,color=INK,align='center')
    if model:
        c.circle(x+w*.12,y+h*.06,9,fill='#F0DED8',outline='#B18D82')
        c.line([(x+w*.12,y+h*.06),points[0]],fill='#AAB4BA',width=2)


def mlp(c,x,y):
    columns=[[(x,y+18),(x,y+57),(x,y+96)],
             [(x+55,y),(x+55,y+29),(x+55,y+58),(x+55,y+87),(x+55,y+116)],
             [(x+111,y+18),(x+111,y+57),(x+111,y+96)]]
    for a,b in zip(columns,columns[1:]):
        for p in a:
            for q in b:c.line([p,q],fill='#CBD6D9',width=1.3)
    for layer in columns:
        for xx,yy in layer:c.circle(xx,yy,9,fill='#CBDFD4',outline=GREEN,width=1.8)


def build():
    c=Canvas(w=2400,h=1640,scale=2,strict_text=True)
    c.text(1200,28,'Evidence-grounded PBPK recommendation',size=48,bold=True,color=KEY,align='center')
    c.text(1200,91,'Traceable knowledge, graph-conditioned components, model matching and mechanistic blueprints',
           size=22,color=GRAY,align='center')
    for x,y,w,h,fill,border,letter,title in [
        (45,150,1050,655,'#EFF5FA','#AEC9DE','a','Evidence & knowledge construction'),
        (1140,150,1215,655,'#F0F6F0','#B9CEBE','b','Evidence-grounded component recommendation'),
        (45,885,1050,675,'#FCF7E9','#DDCA9D','c','Constraint-aware model recommendation'),
        (1140,885,1215,675,'#F5F0F8','#CDBEDD','d','Evidence-grounded blueprint design')]:
        c.rect(x,y,w,h,fill=fill,outline=border,r=17,width=2)
        c.text(x+23,y+20,f'({letter})  {title}',size=32,bold=True,max_width=w-43)
        c.line([(x+22,y+70),(x+w-22,y+70)],fill=border,width=1.4)

    # A: distinct source objects, extraction, typed KG and semantic indexing.
    for xx,name,label in [(83,'paper','Literature'),(203,'bottle','Drug labels'),
                          (327,'pk','PK studies'),(450,'encoder','PBPK models')]:
        c.icon(name,xx,248,s=69,color=BLUE)
        c.text(xx+35,324,label,size=20,align='center')
    c.text(81,382,'Rule-first extraction + LLM assistance',size=26,bold=True)
    c.text(81,425,'Entity / unit normalization',size=22)
    c.text(81,464,'Deduplication + provenance checks',size=22,color=GRAY)
    c.icon('paper',82,514,s=60,color=BLUE)
    c.text(163,525,'Source text',size=26,bold=True)
    c.text(163,558,'Evidence ID + exact source span',size=20,color=GRAY)
    arrow(c,[(535,285),(576,285),(576,408),(630,408)],BLUE)
    nodes={'Drug':(695,407),'Enzyme':(823,294),'Liver':(990,404),
           'Parameter':(816,514),'Source':(1004,578)}
    for a,b in [('Drug','Enzyme'),('Enzyme','Liver'),('Drug','Parameter'),('Source','Parameter'),('Source','Enzyme')]:
        path=[nodes[a],(885,336),(846,294)] if (a,b)==('Source','Enzyme') else [nodes[a],nodes[b]]
        arrow(c,path,'#A6B7C1',1.9,dash=(a=='Source'))
    for i,(name,(xx,yy)) in enumerate(nodes.items()):
        c.circle(xx,yy,21,fill=['#BDD6E9','#C6DDCD','#E6C6B5','#D4C5E0','#E7D8AC'][i],outline=BLUE,width=1.8)
        c.text(xx,yy+27,name,size=20,bold=True,align='center')
    for xx,yy,label in [(650,349,'metabolized by'),(916,345,'expressed in'),(905,546,'supports')]:
        c.rect(xx-4,yy,c.text_width(label,20)+8,24,fill='#EFF5FA',r=3)
        c.text(xx,yy,label,size=20,color=GRAY)
    # Local, sequential calculations belong to the illustrated indexing path.
    arrow(c,[(170,590),(170,615)],BLUE,1.7)
    c.icon('encoder',85,630,s=50,color=BLUE)
    c.text(148,627,'1 Encode source text',size=22,bold=True)
    c.formula(148,670,r'e_g=f(\mathrm{text}_g)',size=MATH,max_width=246)
    arrow(c,[(370,650),(422,650)],BLUE,1.8)
    c.text(445,627,'2 Normalize',size=22,bold=True)
    c.tokens(599,633,n=4,fill='#B9CEDF',size=17)
    c.formula(445,670,r'\hat e_g=e_g/\|e_g\|_2',size=MATH,max_width=248)
    arrow(c,[(707,650),(759,650)],BLUE,1.8)
    c.text(785,627,'3 Index with source ID',size=22,bold=True)
    c.formula(785,670,r'I_g=(\hat e_g,g,E_g)',size=MATH,max_width=270)
    c.text(84,759,'Evidence-linked KG + semantic index + model catalog',size=26,bold=True,color=KEY)
    arrow(c,[(1102,488),(1132,488)],BLUE,3)

    # B: retrieval, heterograph reranking, graph soft tokens and frozen generator.
    c.icon('scientist',1164,244,s=83,color=GREEN)
    c.rect(1264,245,394,84,fill='#FFFFFF',outline='#AAC8B7',r=13,width=2)
    c.text(1284,261,'Drug / population / route',size=22,bold=True)
    c.text(1284,296,'Prediction target + constraints',size=22)
    arrow(c,[(1669,279),(1721,279)],GREEN)
    c.text(1745,243,'Lexical',size=22,bold=True)
    c.text(1745,282,'Dense',size=22,bold=True)
    arrow(c,[(1835,257),(1865,270),(1884,270)],GREEN)
    arrow(c,[(1823,296),(1865,270)],GREEN)
    c.text(1898,251,'Rank fusion',size=26,bold=True)
    c.formula(1696,322,r's_d=\hat e_q^{T}\hat e_g',size=MATH,max_width=184)
    c.formula(1908,302,r's_{\mathrm{RRF}}=\sum_r(60+\mathrm{rank}_r)^{-1}',size=MATH,max_width=411)
    arrow(c,[(2050,270),(2332,270),(2332,365),(1165,365),(1165,477),(1192,477)],GREEN,1.8)
    # A single readable typed graph replaces three copies with tiny labels.
    c.text(1180,373,'Cross-encoder + HGT',size=26,bold=True)
    hg={'Drug':(1305,464),'Enzyme':(1205,430),'Organ':(1401,430),
        'Source':(1205,518),'Parameter':(1401,518)}
    for name,point in hg.items():
        if name!='Drug':c.line([hg['Drug'],point],fill='#9BAFA5',width=2)
    for i,(name,(xx,yy)) in enumerate(hg.items()):
        c.circle(xx,yy,12,fill=['#8EBDB3','#ADCBE1','#D7C2DC','#ECD59C','#F0BCA7'][i],outline=GREEN,width=1.8)
        c.text(xx,yy+17,name,size=20,bold=name=='Drug',align='center')
    c.text(1305,575,'1 Encode subgraphs',size=22,bold=True,align='center')
    arrow(c,[(1431,469),(1484,469)],GREEN)
    # Attention pools query-conditioned subgraph vectors into one global vector.
    for xx,fill in [(1506,'#C4DDD1'),(1551,'#8EBAA2'),(1596,'#BDD5C7')]:
        c.rect(xx,432,26,19,fill=fill,outline=GREEN,r=3,width=1.3)
        c.line([(xx+13,454),(1572,500)],fill='#9BBDA8',width=1.5,arrow=True)
    c.rect(1558,501,28,20,fill='#E7D599',outline=GREEN,r=3,width=1.3)
    c.formula(1602,502,r'h_0',size=MATH,max_width=45)
    c.text(1560,575,'2 Attention pool',size=22,bold=True,align='center')
    # Local vectors bypass pooling; global and local vectors share the projector.
    c.line([(1440,444),(1457,444),(1457,407),(1691,407),(1691,437),(1700,437)],
           fill=GREEN,width=1.5,arrow=True,dash=True)
    c.text(1581,380,'Subgraph vectors',size=20,color=GRAY,align='center')
    arrow(c,[(1630,504),(1660,504),(1695,493)],GREEN)
    mlp(c,1710,419)
    c.text(1765,575,'3 MLP projector',size=22,bold=True,align='center')
    c.tokens(1845,464,n=2,fill=['#E7D599','#B8D8C3'],size=18)
    arrow(c,[(1829,478),(1840,478)],GREEN,1.5)
    arrow(c,[(1894,477),(1920,477)],GREEN)
    c.tokens(1908,409,n=4,fill='#E9C4AC',size=18)
    c.text(1960,380,'Cited evidence',size=20,color=GRAY,align='center')
    arrow(c,[(1950,425),(1974,443)],GREEN,1.5)
    c.icon('robot',1930,435,s=104,color=GREEN)
    c.icon('lock',2020,412,s=34,color=BLUE)
    c.text(1982,575,'4 Frozen LLM',size=22,bold=True,align='center')
    arrow(c,[(2044,486),(2078,486)],GREEN)
    for xx,yy,name,label in [(2100,394,'gut','Absorption'),(2225,394,'blood','Distribution'),
                              (2100,490,'liver','Metabolism'),(2225,490,'kidney','Excretion')]:
        c.icon(name,xx,yy,s=60,color=GREEN)
        c.text(xx+30,yy+64,label,size=20,align='center')
    c.text(2192,589,'Citations + checks',size=20,color=GRAY,align='center')
    # Each short calculation is anchored below the scene that performs it.
    for xx in (1305,1560,1765,1982):
        c.line([(xx,601),(xx,613)],fill=GREEN,width=1.2,dash=True)
    c.formula(1181,628,r'h_i=\mathrm{Enc}_{\mathrm{HGT}}(q,G_i)',size=MATH,max_width=280)
    c.text(1181,675,'Query-conditioned subgraphs',size=20,color=GRAY)
    c.formula(1488,628,r'a_i=\mathrm{softmax}_i(w^{T}h_i)',size=MATH,max_width=232)
    c.formula(1488,673,r'h_0=\sum_i a_i h_i',size=MATH,max_width=232)
    c.formula(1734,628,r't_i=P_\phi(h_i)',size=MATH,max_width=186)
    c.formula(1734,673,r't_0=P_\phi(h_0)',size=MATH,max_width=186)
    c.text(1710,716,'Align → target-token CE',size=20,color=GRAY)
    c.formula(1940,628,r'y=\mathrm{LLM}(q,T,E)',size=MATH,max_width=380)
    c.formula(1940,673,r'T=[t_0,t_1,\ldots]',size=MATH,max_width=380)
    c.text(1183,759,'Train attention + projector; keep the LLM frozen.',size=26,bold=True,color=KEY)
    arrow(c,[(1746,814),(1746,840),(565,840),(565,874)],GREEN,2.5)
    c.rect(757,817,799,45,fill='#FFFFFF',r=6)
    c.text(1155,826,'Component specification + evidence → requirement graph',size=22,color=INK,align='center')

    # C: graph alignment is a subscore, with applicability-first tiered selection.
    c.text(243,973,'Requirement graph',size=26,bold=True,align='center')
    c.text(808,973,'Candidate model graph',size=26,bold=True,align='center')
    typed_graph(c,105,1025,276,156)
    typed_graph(c,670,1025,276,156,True)
    c.text(477,973,'Type affinity',size=26,bold=True,align='center')
    for row in range(4):
        for col in range(4):
            c.rect(412+col*31,1038+row*31,28,28,fill=('#BFCFDB' if row==col else '#F1EBDD'),outline='#DDD6C7',r=3,width=.8)
    arrow(c,[(391,1098),(408,1098)],GOLD)
    arrow(c,[(544,1098),(650,1098)],GOLD)
    c.text(478,1186,'Same-type matches',size=20,color=GRAY,align='center')
    # Decompose the same subscore into type restriction, local match and average.
    for xx,label in [(84,'1 Restrict node types'),(410,'2 Best local match'),(752,'3 Weighted subscore')]:
        c.text(xx,1241,label,size=22,bold=True)
    for xx in (243,477,808):c.line([(xx,1225),(xx,1235)],fill=GOLD,width=1.3,dash=True)
    c.formula(84,1283,r'U_v=\{u\in V_M:\tau(u)=\tau(v)\}',size=MATH,max_width=298)
    c.formula(410,1283,r'm_v=\max_{u\in U_v}\mathrm{sim}(v,u)',size=MATH,max_width=300)
    c.formula(752,1283,r'S_{\mathrm{type}}=\frac{\sum_v w_v m_v}{\sum_v w_v}',size=MATH,max_width=300)
    arrow(c,[(365,1297),(396,1297)],GOLD,1.7)
    arrow(c,[(708,1297),(739,1297)],GOLD,1.7)
    c.text(84,1344,'Absent type → zero contribution',size=20,color=GRAY)
    c.text(410,1344,'Identity / token overlap',size=20,color=GRAY)
    c.text(752,1344,'Eligible nodes; fixed weights',size=20,color=GRAY)
    c.text(83,1380,'Compatibility gates',size=26,bold=True)
    c.text(83,1423,'Route / population / mechanism',size=22,color=GRAY)
    arrow(c,[(440,1412),(592,1412)],GOLD)
    for i,(label,fill) in enumerate([('Reuse / controlled adaptation','#EAF2E9'),('Qualify / reference','#F4EEE1'),('No reliable match → design','#F0E8F5')]):
        c.rect(614,1379+i*46,433,39,fill=fill,outline='#D8CDB4',r=8,width=1)
        c.text(628,1386+i*46,label,size=22,bold=i==2,color=INK)
    c.text(84,1520,'Applicability first; rank within each tier.',size=26,bold=True,color=KEY)
    arrow(c,[(1103,1365),(1132,1365)],GOLD,3)

    # D: controlled model equations, evidence-bound parameters, parallel organs.
    c.text(1168,973,'No reliable reusable model → controlled assembly',size=26,bold=True)
    arrow(c,[(1400,1139),(1284,1069)],PURPLE,2.4)
    arrow(c,[(1198,1080),(1178,1080),(1178,1249),(1197,1249)],PURPLE,2.4)
    c.text(1210,1170,'portal flow',size=20,color=GRAY)
    arrow(c,[(1281,1275),(1387,1179)],PURPLE,2.4)
    arrow(c,[(1387,1151),(1281,1247)],PURPLE,2.4)
    arrow(c,[(1480,1139),(1561,1069)],PURPLE,2.4)
    arrow(c,[(1561,1100),(1493,1163)],PURPLE,2.4)
    arrow(c,[(1490,1179),(1561,1247)],PURPLE,2.4)
    arrow(c,[(1561,1275),(1476,1197)],PURPLE,2.4)
    for name,x,y,label in [('gut',1198,1030,'Gut'),('liver',1198,1230,'Liver'),
                           ('blood',1399,1125,'Blood'),('kidney',1570,1030,'Kidney'),
                           ('tissue',1570,1230,'Tissue')]:
        c.icon(name,x,y,s=80,color=PURPLE)
        c.text(x+40,y+84,label,size=20,bold=True,align='center')
    c.icon('paper',1740,1033,s=72,color=PURPLE)
    c.text(1838,1046,'Evidence-bound parameters',size=26,bold=True,max_width=495)
    c.text(1839,1084,'Controlled templates',size=22,color=GRAY)
    c.rect(1743,1138,555,127,fill='#FFFFFF',outline='#D3C5DF',r=8,width=1.3)
    columns=(1760,1920,2205)
    for xx,label in zip(columns,('Parameter','Context / unit','Source')):
        c.text(xx,1150,label,size=20,bold=True)
    c.line([(1755,1182),(2286,1182)],fill='#D3C5DF',width=1)
    for yy,values in [(1195,('CLint','Evidence-linked','[E1]')),(1232,('V, Kp','Evidence-linked','[E2]'))]:
        for xx,value in zip(columns,values):c.text(xx,yy,value,size=20)
    c.text(1744,1287,'Bind value, unit, context and source',size=20,color=GRAY)
    c.formula(1744,1319,r'\theta_j\leftarrow\mathrm{value}(E_j)',size=MATH,max_width=555)
    # Organ arrows define fluxes; their signed sum defines the solver input.
    for xx,label in [(1170,'1 Sum transport fluxes'),(1545,'2 Balance each compartment'),(1965,'3 Solve & predict')]:
        c.text(xx,1369,label,size=22,bold=True)
    c.line([(1439,1240),(1439,1348),(1270,1348),(1270,1361)],fill=PURPLE,width=1.3,dash=True)
    c.line([(2298,1207),(2320,1207),(2320,1355),(2070,1355),(2070,1361)],fill=PURPLE,width=1.3,dash=True)
    c.formula(1170,1412,r'J_i^+=\sum_j J_{j\to i}',size=MATH,max_width=326)
    c.formula(1170,1457,r'J_i^-=\sum_j J_{i\to j}',size=MATH,max_width=326)
    c.formula(1545,1412,r'\dot A_i=J_i^+-J_i^-+R_i',size=MATH,max_width=362)
    c.formula(1545,1457,r'R_i=R_{\mathrm{in},i}-R_{\mathrm{elim},i}',size=MATH,max_width=362)
    c.formula(1965,1412,r'A(t)=\mathrm{Solve}(\dot A,A(0),\theta)',size=MATH,max_width=354)
    c.formula(1965,1457,r'C_i(t)=A_i(t)/V_i',size=MATH,max_width=354)
    arrow(c,[(1495,1435),(1530,1435)],PURPLE,1.8)
    arrow(c,[(1915,1435),(1950,1435)],PURPLE,1.8)
    c.text(1170,1520,'Evidence-bound blueprints → simulation → independent validation',size=26,bold=True,color=KEY,max_width=1147)

    c.text(56,1590,'q: request   G: evidence subgraph   h: subgraph vector   τ: node type   v: scored node   A: drug amount   J: transport flux   V: volume',
           size=20,color=GRAY)
    c.text(2340,1590,'Method schematic',size=20,color=GRAY,align='right')
    print(c.save(OUT))


if __name__=='__main__':build()
