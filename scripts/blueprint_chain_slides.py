"""Beginner-first explanation of the current PBPK blueprint and its mathematics.

Numerical examples are hypothetical teaching examples; no new experiment is run.
All speaker notes intentionally remain empty.  Details follow the main teaching
sequence so a first-time presenter can explain the workflow before the algebra.
"""
from __future__ import annotations

import build_blueprint_method_slides as st
from presentation_teaching_common import (
    teaching_slide, card, arrow, words, sayline, example, eq, label,
)

TEAL, INK, MUTED = st.TEAL, st.INK, st.MUTED
PALE, WHITE, GREEN, GREEN_PALE = st.PALE, st.WHITE, st.GREEN, st.GREEN_PALE
GOLD, GOLD_PALE = st.GOLD, st.GOLD_PALE


def page(title, step, subtitle='虚拟药 X 贯穿全程 · 数值只用于教学，不是本次实验结果'):
    return teaching_slide(title, subtitle, '蓝图设计 / '+step)


def row(s, y, left, right, *, color=TEAL):
    """A compact, readable glossary or input/output pair."""
    s.rect(50, y, 522, 46, PALE if color == TEAL else GREEN_PALE, None)
    s.text(63, y+9, 185, left, 18, color, bold=True, line=1.15)
    h = s.text(252, y+9, 304, right, 17, INK, line=1.15)
    assert h <= 29, right


def _check(slides):
    for slide in slides:
        assert slide.notes == ''
        for op in slide.ops:
            if op['kind'] in ('equation', 'rect', 'image', 'ellipse'):
                assert op['x'] >= 0 and op['y'] >= 0
                assert op['x']+op['w'] <= st.W+1
                assert op['y']+op['h'] <= st.H+1
            if op['kind'] == 'equation':
                assert op['size'] == st.EQUATION_FONT_SIZE
    return slides


def make_blueprint_chain_slides():
    slides = []

    # 1. Distinguish observations from an existing model, with one continuous task.
    s = page('先定任务：我们已知什么，还想预测什么？', '01')
    card(s, 50, 151, 238, 124, '已知：开发资料', '药物 X，成人 70 kg\n100 mg，输注 1 小时\n少量时间—浓度测量点')
    card(s, 331, 151, 241, 124, '待预测：目标条件', '还是药物 X，同一人体\n50 mg，输注 1 小时\n整条浓度曲线与终点', GREEN, GREEN_PALE)
    arrow(s, (291, 213), (327, 213))
    card(s, 93, 307, 435, 104, '缺的是“可运行的计算模型”', '几次测量是数据；模型是把给药条件\n变成整条浓度曲线的一套方程。', GOLD, GOLD_PALE)
    words(s, [('为什么需要蓝图设计', '推荐阶段没有找到适用模型，但可能已有药物资料和几次测量，我们就用它们构建模型。'),
              ('什么数据先不看', '目标条件的实测浓度先放一边。完成预测后再比较，才能检查模型是否预测准确。')],
          '有几次测量，不等于有模型；我们要把已有资料变成可运行的模型。')
    slides.append(s)

    # 2. Plain-language glossary; no mysterious abbreviations at first contact.
    s = page('输入清单：药物、人体、给药和开发观测', '02')
    row(s, 149, '药物是什么', '名称、分子量、资料来源')
    row(s, 202, '药物有什么性质', '游离比例、亲脂性、酸碱性')
    row(s, 255, '人和给药条件', '年龄、体重；剂量、输注时长')
    row(s, 308, '开发观测是什么', '每个采样时间及对应浓度')
    example(s, '分子量 MW：每摩尔的质量；游离比例 fu：未与蛋白结合的比例。logP / pKa：亲脂性 / 酸碱性。', y=365, h=67)
    words(s, [('输入从哪里来', '人体资料来自研究条件；药物性质来自合格实验或文献。每个值保留单位、来源与是否实测。'),
              ('先做哪些检查', '先核对药物身份和单位，再统一时间与浓度。关键资料缺失时要注明，不能悄悄当作已知。')],
          '我们先整理四类输入，并记清每个数字从哪里来，再开始计算。')
    slides.append(s)

    # 3. Fixed anatomy, with one organ enlarged rather than an unreadable network.
    s = page('第一步：用人体模板搭好“药物运输网络”', '03')
    s.asset('01_human_core.png', 50, 152, 127, 241)
    arrow(s, (182, 258), (230, 258))
    card(s, 239, 151, 327, 72, '各器官由血流连接', '模板给出容积、血流和连接方式')
    card(s, 247, 253, 137, 96, '器官血管区', '随血流\n进出药物')
    card(s, 420, 253, 137, 96, '器官组织区', '接收或\n释放药物', GREEN, GREEN_PALE)
    arrow(s, (388, 301), (416, 301), both=True)
    label(s, '放大一个器官：血管 ↔ 组织', 220, 364, 358)
    words(s, [('人体模板负责什么', '先固定器官容积、血流及连接。15 个器官分成血管区和组织区，再由 3 个血池连接。'),
              ('药物资料怎样变成参数', '亲脂性、游离比例等经分配公式换算为组织的初始等效容量；未知清除等参数随后用开发观测校准。')],
          '人体模板像交通网络；药物参数决定药怎样进出各个器官。')
    slides.append(s)

    # 4. The four effective degrees of freedom and protected evidence.
    s = page('第二步：已知信息保留，未知部分允许调整', '04')
    card(s, 50, 150, 522, 73, '保护有可靠来源的输入', '人体结构；合格净肾清除；实测血液分配等')
    for x, y, title, body in [(50,246,'清除','药物消失多快'),(318,246,'组织容量','组织能容纳多少'),
                               (50,340,'区间交换','进出组织有多快'),(318,340,'血细胞分配','多少进入血细胞')]:
        card(s,x,y,254,77,title,body,GREEN,GREEN_PALE)
    words(s, [('最多调整四个方向', '这里的四个参数是有效调整量。每个只能在允许范围内改变，不能任意修改人体结构。'),
              ('有实测证据时怎么办', '已测且合格的分配值受到保护；不因为拟合不好就覆盖。缺乏实测的参数才按规则开放。')],
          '我们保护可靠的已知信息，只对允许调整的未知量进行校准。')
    slides.append(s)

    # 5. CL is a proportionality coefficient, not blood physically removed.
    s = page('先理解清除：浓度越高，每小时消除越多', '05')
    card(s, 53, 153, 201, 89, '血浆浓度', '1 mg/L')
    card(s, 368, 153, 201, 89, '清除率 CL', '9 L/h', GREEN, GREEN_PALE)
    arrow(s,(258,198),(365,198))
    label(s, '二者相乘', 251, 166, 118)
    eq(s, r'\mathrm{Elimination\ rate}=CL\times C_p', 261, 55)
    label(s, '消除速率（mg/h）＝清除率 × 血浆浓度', 61, 322, 502)
    example(s, '虚拟药 X：9 L/h × 1 mg/L = 每小时消除 9 mg 药物。清除率不是每小时真的抽走 9 L 血。')
    words(s, [('CL 表示什么', 'CL 是清除率，用体积/时间表示。它与浓度相乘，才得到每小时真正被消除的药量。'),
              ('有肾清除证据时', '保留合格的净肾清除，只校准其余清除。当前是全身清除近似，不能据此识别具体代谢酶。')],
          '清除率是一把比例尺；乘上当前浓度，才知道每小时消除多少药。')
    slides.append(s)

    # 6. Explain amount/capacity and difference before a general equation.
    s = page('再理解交换：药从浓度高的地方流向低处', '06')
    card(s,50,151,220,115,'器官血管区','药量 4 mg，容量 2 L\n等效浓度 = 2 mg/L')
    card(s,351,151,220,115,'器官组织区','药量 6 mg，容量 6 L\n等效浓度 = 1 mg/L',GREEN,GREEN_PALE)
    arrow(s,(274,205),(347,205),both=True)
    label(s,'可双向交换',270,174,81,size=14)
    eq(s,r'J=5\times(2-1)=5\ \mathrm{mg/h}',282,51)
    label(s,'交换系数 5 L/h × 浓度差 1 mg/L',59,340,506)
    example(s,'这是一个假设时刻：组织每小时净收到 5 mg，血管同时净失去 5 mg；交换不会凭空造出药。')
    words(s,[('容量在这里是什么意思','用药量除以“等效容量”，得到用于交换的浓度。等效容量反映分配，不等于器官解剖体积。'),
             ('如何知道流向','血管浓度更高就流入组织；组织更高就流回来。本例为血/浆浓度比等于 1 的简化情况。')],
          '两边先把药量换成浓度，再用浓度差乘交换系数，算出药流得多快。')
    slides.append(s)

    # 7. An actual differential equation anchored to physical arrows.
    s = page('微分方程只是在记账：变化＝流入－流出', '07')
    card(s,188,177,233,87,'某器官的血管区','记住这里当前有多少药')
    arrow(s,(54,217),(184,217)); label(s,'血流带入',49,180,130)
    arrow(s,(425,217),(568,217)); label(s,'血流带出',426,180,142)
    arrow(s,(303,266),(303,301),GREEN); label(s,'净流入组织 J',328,273,185,color=GREEN)
    eq(s,r'\frac{dA_v}{dt}=Q(C_{in}-C_{out})-J',313,58)
    example(s,'若血流 Q=10 L/h，流入/流出浓度为 3/2 mg/L，交换 J=5 mg/h：血管药量增加 5 mg/h。',y=374,h=58)
    words(s,[('每个字母怎么读','Av 是血管内药量；左侧是它每小时的变化。Q 是血流，C 是血中浓度，J 是净流向组织的药量。'),
             ('为什么能模拟全身','每个器官都遵守这类记账规则。器官之间的流量接起来，再加入给药与清除，就得到全身方程。')],
          '微分方程并不神秘，就是逐处计算药物每小时增加多少、减少多少。')
    slides.append(s)

    # 8. Calculation input-output; matrix integral stays in appendix.
    s = page('计算机把所有器官一起算，得到整条曲线', '08')
    card(s,50,151,148,96,'一组参数','清除、容量\n交换、分配')
    card(s,233,151,148,96,'联立求解','所有器官\n一起计算')
    card(s,417,151,155,96,'读取血浆','得到每个时刻\n的药物浓度',GREEN,GREEN_PALE)
    arrow(s,(201,193),(229,193)); arrow(s,(384,193),(413,193))
    st.curve(s,93,282,438,104,labels=False)
    label(s,'纵轴：血浆浓度',73,253,191,color=TEAL)
    label(s,'横轴：给药后的时间',367,394,192)
    words(s,[('一次计算需要什么','固定人体网络、一组候选药物参数，以及给药速率。输注 100 mg、持续 1 h，输入就是 100 mg/h。'),
             ('计算得到什么','计算机求解所有药量随时间的变化，再换成血浆浓度。换候选参数时，必须重新计算曲线。')],
          '给定人体、参数和给药条件，方程求解器就能画出一条预测浓度曲线。')
    slides.append(s)

    # 9. Optimization loop with observed dots, readable meaning and one loss.
    s = page('第三步：用开发观测，把未知参数调到合理范围', '09')
    card(s,50,150,151,79,'试一组参数','在允许范围内')
    card(s,235,150,151,79,'计算曲线','与开发点比较')
    card(s,420,150,151,79,'更新参数','保留较优参数',GREEN,GREEN_PALE)
    arrow(s,(204,190),(231,190)); arrow(s,(389,190),(416,190))
    s.line([(496,232),(496,253),(126,253),(126,233)],GOLD,1.6,dash=True,arrow=True)
    st.curve(s,85,279,448,102,dots=True,initial=True,labels=False)
    label(s,'圆点：开发测量；曲线：候选模型计算',70,396,485)
    words(s,[('这一步叫“校准”','算法反复尝试、计算、比较、更新。实际采用有边界的多起点最小二乘，不是神经网络训练。'),
             ('怎样避免随意拟合','只使用开发点，限制参数范围，并抑制过大的分配调整。两种分配公式也只按开发误差选择。')],
          '用开发点调参数；目标研究的观测不能进入这个循环。')
    slides.append(s)

    # 10. Feasibility diagnostic, rather than a generic fit-quality rescue.
    s = page('第四步：先检查“这个范围内能不能做得到”', '10')
    card(s,51,151,521,84,'有时不是没调好，而是范围本身不可能','给药量有限，模型可见浓度也有上限。')
    card(s,51,261,249,100,'范围足够','继续正常校准\n不能随意放宽边界')
    card(s,325,261,247,100,'已证明范围不足','只对未实测的血细胞分配\n按规则放宽允许下界',GOLD,GOLD_PALE)
    example(s,'假设当前范围最高只能达到 2 mg/L，而开发观测为 3 mg/L：先核对资料，再判断是否满足受限修复条件。',y=375,h=57)
    words(s,[('怎么判断不可能','用已给入药量和模型的浓度换算关系，计算可达到的浓度上界。只有明确超界才触发判断。'),
             ('不是为了好看强行修复','实测分配仍受保护。满足条件才扩大未实测参数的搜索范围，然后重做校准；不保证变准。')],
          '先判断原范围能不能解释数据，再决定是否允许修复，而不是误差大就乱改。')
    slides.append(s)

    # 11. Clear lock and target branch, without hidden reference fitting.
    s = page('第五步：锁定模型，再预测新的给药条件', '11')
    st.lock(s,64,162,.78)
    card(s,128,151,441,85,'锁定已选参数与分配公式','记录版本、来源、允许范围与修复理由')
    card(s,50,270,224,95,'输入目标给药','药物 X：50 mg\n输注仍为 1 小时')
    card(s,347,270,224,95,'输出目标曲线','只计算，不再用\n目标浓度修改参数',GREEN,GREEN_PALE)
    arrow(s,(278,316),(343,316))
    example(s,'同一人体、输注时长和参数不变时，这个线性模型的 50 mg 曲线为 100 mg 的一半；非线性药物不应直接套用。',y=376,h=57)
    words(s,[('锁定之后可以改什么','本例只换给药量；药物参数不再变化。改变人群时还需重建人体条件，不能直接照搬全部矩阵。'),
             ('目标实测何时使用','先保存预测，再读取目标实测评分。这保证评分是在检查预测，而不是再次拟合。')],
          '锁定相当于交卷；新条件只进求解器，目标答案只留给最后评分。')
    slides.append(s)

    # 12. First five outputs, curve and scalar distinction.
    s = page('输出一：一条曲线，先读出四个常用数字', '12')
    st.curve(s,92,170,438,124,fill=True,labels=False)
    label(s,'浓度',51,145,65,color=TEAL)
    label(s,'时间',501,303,62)
    s.text(190,152,230,'最高点 → 峰浓度 Cmax',16,TEAL)
    s.text(192,306,329,'峰出现的时刻 → Tmax',16,TEAL)
    s.text(358,236,197,'最后时刻 → Clast',16,GREEN)
    s.text(72,341,489,'曲线下面积 AUC → 一段时间的总暴露',18,GREEN,bold=True)
    example(s,'例如 1–2 h 的浓度从 2 降至 1 mg/L：这段梯形面积 = (2+1)÷2×1 = 1.5 mg·h/L。',y=376,h=56)
    words(s,[('为什么一条曲线够算多项','曲线给出每个时刻的浓度，因此能读取最高值、峰值时刻、末次浓度，以及积分面积。'),
             ('与实测比较时要匹配','预测要按实测时间与窗口比较。稀疏采样中最大的测量值，不一定是人体实际最高峰。')],
          '第一组有五类输出：整条曲线，加上 AUC、Cmax、Tmax 和末次浓度。')
    slides.append(s)

    # 13. Six terminal quantities in a plain-language grid, no equation wall.
    s = page('输出二：尾部可靠时，再计算六个终点', '13')
    boxes=[('半衰期','末端浓度减半要多久'),('AUC∞','含未测尾部的总暴露'),('清除率 CL','由剂量 / 总暴露推得'),
           ('Vz','末端消退对应的分布量'),('MRT','药物平均停留的时间'),('Vss','稳态下的表观分布量')]
    for i,(title,body) in enumerate(boxes):
        card(s,50+(i%2)*267,150+(i//2)*87,255,82,title,body,TEAL if i<3 else GREEN,PALE if i<3 else GREEN_PALE)
    label(s,'Vz、Vss 是“表观体积”，不是某个器官的真实体积。',52,431-19,519,size=15)
    words(s,[('为什么先检查曲线尾部','尾部太短、过于平坦或回归不稳定时，半衰期等派生量可能失真；不合格就标记不可估计。'),
             ('为什么不是把输入抄出来','这些终点从预测轨迹统一计算。构建用的 CL 与终点反算的 CL 区分记录，之后再和参考值比较。')],
          '合起来是 11 类终点；不满足计算条件的项说明原因，不硬填一个数字。')
    slides.append(s)

    # 14. A low-jargon explanation of scoring and its limits.
    s = page('怎样知道预测好不好？用未参与校准的观测评分', '14')
    card(s,50,151,242,96,'例子 1','预测 3，实测 2\n相差 1.5 倍')
    card(s,330,151,242,96,'例子 2','预测 1，实测 2\n相差 2 倍',GREEN,GREEN_PALE)
    eq(s,r'\mathrm{GMFE}=\sqrt{1.5\times2}\approx1.73',262,55)
    label(s,'GMFE：倍数误差的几何平均，越接近 1 越好。',56,333,511,size=17)
    example(s,'上面的数字仅是评分教学例子。真正评价要逐药汇总，并同时看终点、失败和不确定性。')
    words(s,[('倍数误差如何理解','预测是实测的 2 倍，或者只有一半，都记为 2 倍误差。完全一致时是 1，不是 0。'),
             ('一个数字不能代表全部','曲线与每个终点分别评分。还要看是否存在明显退步、尾部不可靠和来源差异，不合成万能总分。')],
          '开发点负责调参，目标点负责打分；误差越接近 1，数值越一致。')
    slides.append(s)

    # 15. Make 4+1 optional and prevent the main-experiment misunderstanding.
    s = page('“4＋1”是补充观测的实验，不是蓝图定义', '15')
    card(s,50,151,227,96,'第一次构建','4 个已有开发点\n→ 校准 → 预测')
    card(s,344,151,227,96,'补一个开发点','按时间覆盖规则选择\n→ 重新校准 → 再预测',GREEN,GREEN_PALE)
    arrow(s,(281,193),(340,193))
    card(s,90,285,442,112,'检验的问题：多补一个点有没有实际价值？','比较同一目标研究前后的预测误差。\n目标实测始终只用于评分。',GOLD,GOLD_PALE)
    words(s,[('直接给 5 个点行不行','可以。已有 5 个点就直接校准；有更多合格开发点也能使用，不必人为先扣掉一个。'),
             ('当前选点采用什么','采用简单时间覆盖规则。现有结果不支持复杂学习选点器普遍更优，因此它不是必要模块。')],
          '主算法是构建与预测；4＋1 只是检验补充开发观测有多少帮助。')
    slides.append(s)

    # 16. A defensible contribution, not stronger claims than evidence supports.
    s = page('我们的贡献：让构建过程自动、受约束、可追溯', '16')
    card(s,50,150,251,100,'有证据：保护','保留来源和单位\n不覆盖合格实测')
    card(s,321,150,251,100,'有未知：受限估计','在允许范围内校准\n留下选择理由',GREEN,GREEN_PALE)
    card(s,50,276,251,100,'有问题：检查再修复','判断范围是否可行\n只开放允许修改的部分',GOLD,GOLD_PALE)
    card(s,321,276,251,100,'有输出：标明边界','锁定后再预测\n报告质量与不可支持项')
    words(s,[('创新具体落在哪里','把药物证据、受限校准、可行性检查和预测质量连成可复现流程；并不是重新发明人体方程。'),
             ('需要守住的结论边界','当前用本药观测做单次静脉血浆预测；不证明口服、器官浓度或未知药都准确，也不提供临床安全保证。')],
          '输入资料与需求，输出可运行模型、预测结果，以及为什么这样构建和哪些还不可靠。')
    slides.append(s)
    assert len(slides) == 16
    return _check(slides)


def make_blueprint_technical_slides():
    """Five optional equation pages, after the beginner-first main narrative."""
    slides=[]

    s=page('技术备查 1：四个调整量如何变成模型参数','公式 1/5','初始容量：PK-Sim Standard / Rodgers–Rowland；两者分别校准，再按开发损失选择。')
    eq(s,r'CL=CL_R+s_{CL}(CL_0-CL_R)',151,48)
    eq(s,r'K_{ie}=s_KK_{ie,0},\quad PS_i=s_{ex}Q_{i,eff}',218,48)
    eq(s,r'K_{RBC}=s_{RBC}K_{RBC,0}',285,45)
    eq(s,r'R_{BP}=1-Hct+Hct\,K_{RBC}',348,45)
    label(s,'下标 0：基础值；s：允许调整的倍率；i：第 i 个器官。',55,412,514,size=15)
    words(s,[('清除与组织轴','CL 为总清除；CL_R 为保留的净肾清除。Kie 为组织等效容量；PS 为按有效血流 Q 调整的交换系数。'),
             ('血细胞轴','K_RBC 为血细胞/血浆分配比；Hct 为血细胞比容；R_BP 为血/浆浓度比，随血细胞分配一起更新。')],
          '这些映射将少量可调量，转换成每次方程求解真正需要的参数。')
    slides.append(s)

    s=page('技术备查 2：一个器官的交换与药量方程','公式 2/5','普通器官的示意方程；门静脉等多入口关系在完整网络中保留')
    eq(s,r'K_{iv}=V_{i,pl}+K_{RBC}V_{i,RBC}',151,46)
    eq(s,r'J_i=PS_iR_{BP}\left(\frac{A_{iv}}{K_{iv}}-\frac{A_{ie}}{K_{ie}}\right)',213,64)
    eq(s,r'\frac{dA_{iv}}{dt}=Q_i(C_{a,b}-C_{iv,b})-J_i',286,59)
    eq(s,r'\frac{dA_{ie}}{dt}=J_i',366,53)
    words(s,[('血管区与组织区','A 是药量，v/e 表示血管/组织；K 是等效容量；V 是解剖容积；pl/RBC 表示血浆/血细胞。'),
             ('浓度和流量','Ca,b 是流入动脉血浓度；Civ,b 是流出血浓度。Q 是血流；J 在两区等量增减，保证交换守恒。')],
          '组织交换只移动药；全身总药量的增减来自给药和清除。')
    slides.append(s)

    s=page('技术备查 3：全身线性系统如何得到血浆曲线','公式 3/5','15 器官 × 双区＋3 血池＝33 个体内药量状态；另记累计消除量')
    eq(s,r'\dot{\mathbf{A}}=\mathbf{M}\mathbf{A}+\mathbf{b}\,r(t)',147,52)
    eq(s,r'r(t)=D/T\ (0\leq t\leq T),\quad 0\ (t>T)',210,43)
    eq(s,r'\mathbf{A}(t)=\int_0^{\min(t,T)} e^{\mathbf{M}(t-\tau)}\mathbf{b}\frac{D}{T}\,d\tau',268,67)
    eq(s,r'\widehat C_p(t)=\mathbf{h}^{\mathsf{T}}\mathbf{A}(t)',354,49)
    words(s,[('矩阵和输入各是什么','A 是各处药量；M 装入血流、容量、交换和清除；b 指定给药入口。D 是剂量，T 是输注时长。'),
             ('怎样读出浓度','从初始药量为零开始，用矩阵指数计算。h 把血液药量换成采样血浆浓度；分配变了，h 也更新。')],
          '它是从人体参数派生的线性方程系统，不是完整原生 PK-Sim 的所有机制。')
    slides.append(s)

    s=page('技术备查 4：校准目标与浓度可行性上界','公式 4/5','实际算法：多起点、有边界的最小二乘；不是蓝图神经网络')
    eq(s,r'L=\frac{1}{n}\sum_j\left[\ln\frac{\widehat C_p(t_j)}{C_j}\right]^2+0.01R',146,67)
    eq(s,r'R=\left(\frac{\ln s_K}{\ln10}\right)^2+\left(\frac{\ln s_{ex}}{\ln100}\right)^2',225,61)
    eq(s,r'\qquad+\left(\frac{\ln s_{RBC}}{\ln10}\right)^2',294,61)
    eq(s,r'C_p(t)\leq D_{delivered}(t)\max_i h_i',369,47)
    words(s,[('开发损失如何计算','Cj 是第 j 个开发观测；预测在同一时间 tj 比较；n 是点数。R 抑制过大分配调整，不对 CL 加此项。'),
             ('上界只说明可行性','已给入药量乘最大的浓度换算系数，得到必要上界。原范围超界且分配未实测，才按规则考虑修复。')],
          '拟合更好不等于参数唯一；同样好拟合的参数也可能产生不同外推。')
    slides.append(s)

    s=page('技术备查 5：尾部终点如何从曲线统一计算','公式 5/5','静脉单次给药/输注语境；质量检查通过后才计算，缺项不强填')
    eq(s,r't_{1/2}=\frac{\ln2}{\lambda_z},\quad AUC_\infty=AUC_{0-t}+\frac{C_{last}}{\lambda_z}',146,69)
    eq(s,r'CL_{NCA}=\frac{D}{AUC_\infty},\quad V_z=\frac{CL_{NCA}}{\lambda_z}',234,61)
    eq(s,r'MRT=\frac{AUMC_\infty}{AUC_\infty}-\frac{T}{2}',313,53)
    eq(s,r'V_{ss}=CL_{NCA}\,MRT',382,43)
    words(s,[('尾部与面积','λz 是末端 ln(浓度) 回归的负斜率；AUMC 是时间加权浓度面积；T 是输注时长，D 是剂量。'),
             ('不要混淆输入和输出','CL_NCA 是由轨迹反算的清除；Vz、Vss 是表观分布体积。末端回归或外推不合格时不报告派生终点。')],
          '先验证尾部质量，再算派生终点；不能把不稳定尾部包装成可靠半衰期。')
    slides.append(s)
    assert len(slides)==5
    return _check(slides)


if __name__ == '__main__':
    main=make_blueprint_chain_slides()
    technical=make_blueprint_technical_slides()
    print({'beginner_blueprint':len(main),'technical_backup':len(technical),'speaker_notes':'empty'})
