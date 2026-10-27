"""Shared, spacious layout for an introductory PBPK presentation."""
from build_blueprint_method_slides import *


def teaching_slide(title, subtitle, section):
    s = Slide(title, subtitle, section, '')
    s.ops[0]['rows'] = ['PBPK · 从零理解   /   ' + section]
    s.ops[4]['rows'] = ['PBPK · 从输入到预测 · 图示与虚拟药数值用于教学，非新增实验']
    return s


def card(s, x, y, w, h, title, body='', color=TEAL, fill=PALE):
    s.rect(x, y, w, h, fill, color, .85)
    title_h = s.text(x+12, y+10, w-24, title, 19, color,
                     bold=True, align='center', line=1.15)
    if body:
        body_h = s.text(x+12, y+title_h+16, w-24, body, 17, INK,
                        align='center', line=1.24)
        assert title_h+16+body_h+5 <= h, 'Card too short: '+title


def arrow(s, a, b, color=TEAL, dash=False, both=False):
    s.line([a, b], color, 1.8, dash=dash, arrow=True, both=both)


def words(s, blocks, say):
    """Two short explanations and an on-slide sentence the presenter can use."""
    assert len(blocks) == 2
    s.line([(590, 145), (590, 431)], 'DCE5E8', 1)
    for y, (title, body) in zip((153, 292), blocks):
        th = s.text(614, y, 302, title, 20, TEAL, bold=True, line=1.16)
        assert th <= 28, 'Shorten explanation heading: '+title
        bh = s.text(614, y+36, 302, body, 18, INK, line=1.28)
        assert bh <= 97, 'Shorten explanation body: '+body
    sayline(s, say)


def sayline(s, say):
    s.rect(42, 451, 876, 44, PALE, None)
    h = s.text(55, 456, 850, '这页这样讲：'+say, 16, TEAL, line=1.15)
    assert h <= 40, 'Shorten spoken summary: '+say


def example(s, text, y=371, h=60):
    s.rect(50, y, 522, h, GOLD_PALE, None)
    used = s.text(63, y+9, 496, text, 16, GOLD, line=1.2)
    assert used <= h-13, 'Shorten example: '+text


def eq(s, latex, y, h=52, x=52, w=520, color=INK):
    s.eq(x, y, w, h, latex, EQUATION_FONT_SIZE, color)


def label(s, text, x, y, w, color=MUTED, size=16):
    s.text(x, y, w, text, size, color, align='center', line=1.2)


def teaching_curve(s, x=88, y=185, w=425, h=170, *, dots=False, fill=False):
    curve(s, x, y, w, h, dots=dots, fill=fill, labels=False)
    s.text(x-22, y-34, w, '血浆浓度（每升血浆里有多少药）', 16, TEAL)
    s.text(x+w-112, y+h+8, 130, '给药后的时间', 16, MUTED)
    if dots:
        s.ellipse(x+2, y+h+41, 8, fill=WHITE, stroke=GOLD, width=1.7)
        s.text(x+16, y+h+34, 210, '圆点：假设的测量值', 15, GOLD)
    return s


def make_foundation_slides():
    pages = []
    s = teaching_slide('先问一个问题：给药后，体内会有多少药？',
                       '零基础路线：认识药代动力学，理解蓝图构建，再看校准、预测与验证。', '入门 01')
    card(s, 57, 153, 504, 65, '问题：换个剂量或输注时长，浓度会怎样变？')
    infusion(s, 83, 262, .9)
    s.asset('01_human_core.png', 242, 236, 92, 154)
    curve(s, 407, 267, 137, 88, labels=False)
    arrow(s, (143, 305), (223, 305))
    arrow(s, (339, 305), (395, 305))
    label(s, '输入：怎样给药', 48, 395, 165, TEAL)
    label(s, '过程：药在体内移动', 215, 395, 180, GREEN)
    label(s, '输出：浓度随时间变', 398, 395, 178, TEAL)
    words(s, [('什么是 PK', 'PK 是药代动力学：研究药物进入人体后，怎样分布、被清除，浓度怎样随时间变化。'),
              ('我们想解决什么', '没有适用模型时，利用人体模板、药物资料与少量观测，构建模型并预测新的给药条件。')],
          '我们想回答的不是“药有没有用”，而是“这样给药后，体内浓度怎样变化”。')
    pages.append(s)

    s = teaching_slide('先学会读这条线，后面就容易理解了',
                       '曲线图是示意；圆点不是本次新采集的临床数据。', '入门 02')
    teaching_curve(s, x=81, y=194, w=439, h=140, dots=True)
    label(s, '实线：模型算出的浓度', 228, 383, 301, TEAL)
    words(s, [('横轴和纵轴是什么', '横轴是给药后的时间；纵轴是血浆浓度。血浆是去掉血细胞后的血液液体部分。'),
              ('线和点各代表什么', '线是模型预测；点来自采血测量。后面先用一组点调整模型，再用另一组点检查预测。')],
          '看图先分清：点是测出来的，线是算出来的；检验预测要用没参与调参的观测。')
    pages.append(s)

    s = teaching_slide('药量和浓度不同：一个是总数，一个是密度',
                       '先认三个单位：mg 是药量，L 是体积，h 是时间。', '入门 03')
    for x, vol, height, color in [(91, '1 L', 76, TEAL), (339, '2 L', 111, GREEN)]:
        s.rect(x, 169, 144, 139, WHITE, color, 1.2, rounded=False)
        s.rect(x+2, 308-height, 140, height-2, PALE if color==TEAL else GREEN_PALE, None, rounded=False)
        for dx,dy in [(31,30),(70,30),(31,58),(70,58)]:
            s.ellipse(x+dx, 232+dy, 9, fill=GOLD, stroke=None)
        label(s, '同样 4 mg 药', x-5, 144, 154, color)
        label(s, vol+' → '+('4' if vol=='1 L' else '2')+' mg/L', x-8, 317, 160, color)
    eq(s, r'C=\frac{A}{V}', 365, 54)
    words(s, [('同样药量可以有不同浓度', '4 mg 药放在 1 L 液体中，浓度是 4 mg/L；放在 2 L 中，就是 2 mg/L。'),
              ('读懂最简单的公式', 'A 是药量，V 是液体体积，C 是浓度。人体还有组织结合与分配，后面会用“等效容量”换算。')],
          '药量是有多少药，浓度是每升有多少药；PBPK 先计算各处药量，再换算浓度。')
    pages.append(s)

    s = teaching_slide('PBPK：用器官和血流搭一张计算地图',
                       'PBPK 的全称是“生理药代动力学”：在 PK 中显式利用人体生理结构。', '入门 04')
    card(s, 72, 158, 477, 53, '血液循环：把药物送往器官，再带回来')
    for x,title,body,col,fill in [(55,'肝','一个计算区域',TEAL,PALE),
                                 (231,'肾','一个计算区域',GREEN,GREEN_PALE),
                                 (407,'其他组织','各自记录药量',GOLD,GOLD_PALE)]:
        card(s,x,279,151,102,title,body,col,fill)
        arrow(s,(x+75,215),(x+75,273),col,both=True)
    label(s,'图中只画三个区域；真实程序会同时连接更多器官。',56,401,504)
    words(s, [('器官为什么画成方框', '方框是计算单元，分别记录药量；箭头表示血流或药物交换。它不是人体里真的有一个盒子。'),
              ('和两室模型有什么不同', '两室模型把身体概括成两个有效空间；PBPK 按生理结构划分。结构更细，不代表一定预测更准。')],
          'PBPK 的底座是一张有生理依据的计算地图：每个区域记药量，区域之间算流动。')
    pages.append(s)

    s = teaching_slide('整个项目分三步，每一步交付不同的东西',
                       '先理解三部分的分工，再进入“没有适用模型时怎么办”。', '入门 05')
    card(s,57,153,503,62,'① 组件推荐：找有证据支持的机制')
    arrow(s,(308,219),(308,240))
    card(s,57,245,503,62,'② 模型推荐：检查有没有适用的现成模型',color=GREEN,fill=GREEN_PALE)
    arrow(s,(200,312),(170,348),GREEN)
    arrow(s,(419,312),(440,348),GOLD)
    card(s,52,355,221,67,'有：核验并复用',color=GREEN,fill=GREEN_PALE)
    card(s,310,355,255,67,'无：③ 进入蓝图设计',color=GOLD,fill=GOLD_PALE)
    words(s, [('前两步负责找和选', '组件推荐输出“考虑什么机制及其证据”；模型推荐输出“哪些已有模型适用”。二者都不是浓度曲线。'),
              ('蓝图负责构建和计算', '没有可复用模型时，将药物资料放入人体框架，估计允许调整的参数，生成曲线并检查预测。')],
          '找机制、选模型、建模型是三件事；没有合适的成品，才进入蓝图构建。')
    pages.append(s)
    return pages


def make_teaching_conclusion_slides():
    pages = []
    s = teaching_slide('当前结果说明什么？先看数字，再看边界',
                       '下面是已有 30 药回顾性实验；本次只是重新讲解，没有重新跑实验。', '理解结果')
    card(s,55,150,507,59,'相同五个开发浓度点 → 比较目标研究')
    columns=[(69,192,'评价项'),(275,124,'蓝图'),(425,120,'两室模型')]
    for x,w,t in columns:
        s.text(x,237,w,t,18,TEAL,bold=True,align='center')
    s.line([(61,271),(558,271)],LINE,1)
    for y,name,a,b in [(286,'采样峰值 GMFE','1.465','1.809'),(342,'整条曲线 GMFE','1.553','1.703')]:
        s.rect(61,y-3,497,47,PALE,None,rounded=False)
        for x,w,t in [(69,192,name),(275,124,a),(425,120,b)]:
            s.text(x,y+7,w,t,18,TEAL if t==a else INK,align='center')
    label(s,'同样五点 ≠ 全部先验信息相同；两室没有完整人体结构。',55,403,512,size=15)
    words(s,[('怎样读这些数值', 'GMFE 越接近 1 越好。本批平均采样峰值误差较低；采样峰值不一定等于真实连续峰值。'),
             ('不能过度推论', '曲线的配对区间仍包含无差异；有些药会退步。不能说所有药、所有终点都胜出，更不能保证临床安全。')],
          '目前证据支持有限条件下的收益；必须同时说明药物范围、信息条件和未解决的问题。')
    pages.append(s)

    s=teaching_slide('最后，用这四句话把蓝图设计讲清楚',
                     '正文到这里结束。后面的备查页解释参数映射、微分方程、校准和终点计算。', '一分钟串讲')
    rows=[('01','整理药物、人体、给药条件和开发观测，核对来源与单位。',TEAL,PALE),
          ('02','用人体模板和药物资料组装方程，保留合格实测参数。',GREEN,GREEN_PALE),
          ('03','用开发观测校准未知量，检查参数范围是否需要受限修复。',TEAL,PALE),
          ('04','锁定模型，计算新条件的曲线和终点，再用独立观测评分。',GOLD,GOLD_PALE)]
    for k,(num,text,col,fill) in enumerate(rows):
        y=147+k*72
        s.rect(50,y,864,60,fill,None)
        s.text(66,y+12,61,num,24,col,bold=True)
        h=s.text(138,y+14,752,text,21,INK,line=1.15)
        assert h<46
    sayline(s,'核心是让“怎样建、哪些能改、怎样验证”都有依据；不是只画一条看起来合理的曲线。')
    pages.append(s)
    return pages
