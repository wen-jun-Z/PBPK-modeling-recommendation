"""Introductory model recommendation: two main slides and one backup."""
from presentation_teaching_common import *


def make_model_slides():
    pages = []
    s = teaching_slide(
        'Models: can an existing model serve this task?',
        'An existing PBPK model is an executable whole. Recommendation selects it without refitting it.',
        'Models / Purpose')
    card(s, 52, 155, 224, 96, 'User requirements', 'Drug, population, dose,\nroute, and endpoints')
    card(s, 334, 155, 230, 96, 'Candidate information', 'Supported conditions\nand existing validation', GREEN, GREEN_PALE)
    arrow(s, (164, 252), (253, 274))
    arrow(s, (449, 252), (362, 274), GREEN)
    card(s, 169, 282, 276, 57, 'Match needs to capabilities')
    arrow(s, (270, 343), (172, 366))
    arrow(s, (344, 343), (448, 366), GOLD)
    card(s, 52, 372, 235, 55, 'Suitable → verify and reuse')
    card(s, 327, 372, 237, 55, 'None suitable → blueprint', color=GOLD, fill=GOLD_PALE)
    words(s, [
        ('Check the match', 'A shared drug name is insufficient. Population, dosing, dose level, and required endpoints must fit the supported scope.'),
        ('Scores are not concentrations', 'Ranking prioritizes candidates for review. Reuse still requires source, execution, and validation checks.'),
    ], 'Verify and reuse a suitable model; otherwise continue to blueprint design.')
    pages.append(s)

    s = teaching_slide(
        'A small neural network learns candidate matching',
        'Encode text and conditions as numbers, then compare them; no drug concentration is calculated here.',
        'Models / Learning')
    card(s, 53, 151, 510, 62, 'Request text + candidate description + explicit conditions')
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
    label(s, 'Text and condition vectors', 45, 371, 185, size=15)
    label(s, 'MLP: a small neural network', 218, 371, 173, GREEN, 15)
    label(s, 'Match score', 422, 371, 129, GOLD, 15)
    eq(s, r's=f_{\theta}(\mathbf{x})', 397, h=35, x=54, w=215)
    label(s, 'x: inputs; θ: learned weights; s: ranking score', 278, 398, 286, size=15)
    words(s, [
        ('How is text handled?', 'An encoder represents text as numerical embeddings. Explicit drug and dose conditions also become inputs to the network.'),
        ('What is learned?', 'Historical requests and match labels teach comparisons. New requests trigger candidate scoring, not drug-model retraining.'),
    ], 'The MLP learns suitability; the blueprint later calculates concentrations with differential equations.')
    pages.append(s)
    assert len(pages) == 2
    return pages


def make_model_technical_slides():
    s = teaching_slide(
        'Technical detail: combining text and conditions',
        'The Qwen3-Embedding-0.6B encoder is frozen; training updates the downstream network of about 224,000 parameters.',
        'Models / Technical')
    label(s, 'q, c: 128 dimensions each; f: 204 condition features', 52, 154, 520, size=17)
    eq(s, r'\mathbf{x}=[\mathbf{q};\mathbf{c};|\mathbf{q}-\mathbf{c}|;\mathbf{q}\odot\mathbf{c};\mathbf{f}]', 192, h=46)
    label(s, 'Concatenation: 128 × 4 + 204 = 716 dimensions', 52, 242, 520, size=17)
    eq(s, r'\mathbf{h}=\mathrm{GELU}(\mathrm{LN}(W\mathbf{x}+\mathbf{b}))', 265, h=43)
    label(s, 'h: fused 128-dimensional representation', 52, 312, 520, size=15)
    eq(s, r's=\mathbf{w}_s^{\mathsf{T}}\mathbf{h}+b_s', 342, h=36)
    card(s, 62, 384, 500, 45, 'One representation → ranking, support, and applicability')
    words(s, [
        ('What do the steps do?', 'Text shrinks from 1024 to 128 dimensions. W and b are learned; LN stabilizes values, GELU adds nonlinearity, and dropout masks units.'),
        ('How is it evaluated?', 'Hold out drug and shared-source groups. Better ranking is not better PK prediction; few curve cases cannot establish overall gains.'),
    ], '⊙ denotes elementwise multiplication. Score s determines rank; the other heads aid learning and interpretation.')
    return [s]
