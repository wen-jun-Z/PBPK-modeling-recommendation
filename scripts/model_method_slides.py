"""Introductory model recommendation: two main slides and one backup."""
from presentation_teaching_common import *


def make_model_slides():
    pages = []
    s = teaching_slide(
        '模型推荐：看看已有模型能不能直接用于当前任务',
        '已有 PBPK 模型是一个能计算的整体；模型推荐负责选它，不负责重新拟合它。',
        '模型推荐 / 是什么')
    card(s, 52, 155, 224, 96, '用户的要求', '药物、人群、剂量\n给药方式、预测终点')
    card(s, 334, 155, 230, 96, '已有模型资料', '支持什么条件\n已有怎样的验证', GREEN, GREEN_PALE)
    arrow(s, (164, 252), (253, 274))
    arrow(s, (449, 252), (362, 274), GREEN)
    card(s, 169, 282, 276, 57, '比较要求与模型能力')
    arrow(s, (270, 343), (172, 366))
    arrow(s, (344, 343), (448, 366), GOLD)
    card(s, 52, 372, 235, 55, '适用 → 核验复用')
    card(s, 327, 372, 237, 55, '无适用 → 蓝图设计', color=GOLD, fill=GOLD_PALE)
    words(s, [
        ('先比较是否匹配', '药名相同还不够，还要看人群、给药方式、剂量和需要的终点是否在候选的支持范围内。'),
        ('推荐分数不是浓度', '排序得到的是优先检查的候选。是否能复用，还要核对资料、执行条件和验证依据。'),
    ], '有合适的模型就核验复用，没有才进入后面的蓝图设计。')
    pages.append(s)

    s = teaching_slide(
        '小型神经网络学的是“哪个候选更匹配”',
        '先把文字和条件变成数字，再让网络比较；这一页没有计算药物浓度。',
        '模型推荐 / 怎么学')
    card(s, 53, 151, 510, 62, '请求文字＋候选说明＋明确条件')
    arrow(s, (308, 218), (308, 243))
    input_points = [(106, y) for y in (266, 295, 324, 353)]
    hidden_points = [(297, y) for y in (280, 310, 340)]
    output_point = (487, 310)
    for a in input_points:
        for b in hidden_points:
            s.line([a, b], 'D5E3E8', 1)
    for a in hidden_points:
        s.line([a, output_point], 'BDD6DF', 1.2)
    for x, y in input_points:
        s.ellipse(x-9, y-9, 18, fill=PALE, stroke=TEAL, width=1.2)
    for x, y in hidden_points:
        s.ellipse(x-10, y-10, 20, fill=GREEN_PALE, stroke=GREEN, width=1.2)
    s.ellipse(output_point[0]-12, output_point[1]-12, 24, fill=GOLD_PALE, stroke=GOLD, width=1.2)
    label(s, '文字与条件的数字表示', 45, 371, 185, size=15)
    label(s, 'MLP：小型神经网络', 218, 371, 173, GREEN, 15)
    label(s, '匹配分数', 422, 371, 129, GOLD, 15)
    eq(s, r's=f_{\theta}(\mathbf{x})', 397, h=35, x=54, w=215)
    label(s, 'x 是输入数字；θ 是训练学到的权重；s 用于排序', 278, 398, 286, size=15)
    words(s, [
        ('怎样处理文字', '编码器把一段文字表示成一组数字，这叫嵌入。药物、剂量等明确条件也变成数字，一起交给网络。'),
        ('训练学到了什么', '网络从历史请求和匹配标签中学习比较规则。新请求到来时，它给候选打分，而不是重新训练一个药物模型。'),
    ], 'MLP 学的是谁更匹配；后面的蓝图用微分方程计算浓度，两者职责不同。')
    pages.append(s)
    assert len(pages) == 2
    return pages


def make_model_technical_slides():
    s = teaching_slide(
        '技术补充：匹配网络如何合并文字与条件',
        'Qwen3-Embedding-0.6B 编码器冻结；训练更新的是后面的约 22.4 万参数的小网络。',
        '模型推荐 / 技术补充')
    label(s, 'q、c：请求和候选各 128 维；f：204 维条件特征', 52, 154, 520, size=17)
    eq(s, r'\mathbf{x}=[\mathbf{q};\mathbf{c};|\mathbf{q}-\mathbf{c}|;\mathbf{q}\odot\mathbf{c};\mathbf{f}]', 192, h=46)
    label(s, '拼接：128 × 4＋204＝716 维', 52, 242, 520, size=17)
    eq(s, r'\mathbf{h}=\mathrm{GELU}(\mathrm{LN}(W\mathbf{x}+\mathbf{b}))', 265, h=43)
    label(s, 'h：融合后的 128 维表示', 52, 312, 520, size=15)
    eq(s, r's=\mathbf{w}_s^{\mathsf{T}}\mathbf{h}+b_s', 342, h=36)
    card(s, 62, 384, 500, 45, '同一表示 → 排序、支持等级、适用性三个输出')
    words(s, [
        ('每一步在做什么', '1024 维文字先压到 128 维。W、b 是学到的权重和偏置；LN 稳定数值，GELU 加入非线性；训练还会随机屏蔽部分单元。'),
        ('结果怎样验证', '按药物和共享来源分组留出，减少同源交叉。排序改善不等于浓度更准；当前少量曲线案例不足以证明整体预测收益。'),
    ], '图中的 ⊙ 是逐项相乘；排名由分数 s 决定，另外两个输出帮助学习和解释。')
    return [s]
