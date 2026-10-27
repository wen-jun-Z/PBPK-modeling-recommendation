"""Introductory component recommendation: two main slides and two backups."""
from presentation_teaching_common import *


def make_component_slides():
    pages = []
    s = teaching_slide(
        'Components: choose the processes a model needs',
        'Components form parts of a model; calculating concentration requires their correct assembly.',
        'Components / Purpose')
    card(s, 64, 151, 492, 67, 'Drug X: predict plasma concentration after IV infusion')
    arrow(s, (310, 222), (310, 239))
    s.line([(130, 242), (490, 242)], TEAL, 1.3)
    for x, title, body, color, fill in [
        (50, 'Input', 'Drug enters\nduring infusion', TEAL, PALE),
        (230, 'Distribution', 'Binding affects\ndrug distribution', GREEN, GREEN_PALE),
        (410, 'Elimination', 'Drug is metabolized\nor excreted', GOLD, GOLD_PALE),
    ]:
        arrow(s, (x+80, 242), (x+80, 256), color)
        card(s, x, 261, 160, 103, title, body, color, fill)
    example(s, 'Illustrative processes only; actual recommendations depend on drug evidence and intended use.')
    words(s, [
        ('What is a component?', 'A process within a model, such as drug entry, distribution, or elimination.'),
        ('What is delivered?', 'Recommended processes, supporting evidence, and gaps. This stage does not generate a concentration curve.'),
    ], 'Components define which processes to consider; the assembled model calculates concentrations.')
    pages.append(s)

    s = teaching_slide(
        'Retrieve evidence, then check its relevance',
        'A drug-related source may not support the requested population, administration route, or prediction endpoint.',
        'Components / Workflow')
    card(s, 54, 157, 226, 96, '① Retrieve', 'Drug names, keywords,\nand similar meanings')
    card(s, 330, 157, 226, 96, '② Link evidence', 'Drug–mechanism–source\nrelations in one graph', GREEN, GREEN_PALE)
    arrow(s, (286, 202), (324, 202))
    arrow(s, (443, 254), (443, 292), GREEN)
    card(s, 330, 299, 226, 96, '③ Check use', 'Population, route,\nand prediction goal', GOLD, GOLD_PALE)
    card(s, 54, 299, 226, 96, '④ Recommend', 'Components, citations,\nand traceable gaps')
    arrow(s, (324, 344), (286, 344), GOLD)
    label(s, 'Unsupported items remain explicit evidence gaps', 54, 410, 502)
    words(s, [
        ('Find relevant material', 'Use keyword and semantic retrieval. A graph links drugs, mechanisms, and sources to clarify what each source supports.'),
        ('Check applicability', 'Generate recommendations after checking conditions and citations. Unstated facts are not treated as supported.'),
    ], 'Every proposed mechanism needs supporting evidence and applicable conditions.')
    pages.append(s)
    assert len(pages) == 2
    return pages


def make_component_technical_slides():
    pages = []
    s = teaching_slide(
        'Technical detail: text and graph evidence matching',
        'Retrieve broadly, then rerank text and graph relations for the requested use.',
        'Components / Technical 1')
    card(s, 55, 151, 502, 71, 'Keywords + five semantic encoders', '50 subgraphs per retriever → rank fusion to 100')
    arrow(s, (307, 225), (307, 242))
    s.line([(176, 246), (437, 246)], TEAL, 1.3)
    arrow(s, (176, 246), (176, 259))
    arrow(s, (437, 246), (437, 259), GREEN)
    card(s, 55, 264, 242, 96, 'Text reranking', 'Qwen3-Reranker\n0.6B')
    card(s, 316, 264, 241, 96, 'Graph matching', '2 layers, 4 heads\n256-dimensional vectors', GREEN, GREEN_PALE)
    arrow(s, (176, 364), (245, 376))
    arrow(s, (437, 364), (365, 376), GREEN)
    card(s, 94, 380, 424, 47, 'Check conditions, deduplicate → up to 8 subgraphs')
    words(s, [
        ('What is HGT?', 'A graph neural network that distinguishes node and relation types. Two layers pass information; four attention heads run in parallel.'),
        ('What is an embedding?', 'Text becomes a comparable vector. Five encoders include Qwen3-Embedding-0.6B; graph matching also uses the current request.'),
    ], 'The full graph matcher includes teacher and residual mechanisms; its gains cannot all be attributed to HGT.')
    pages.append(s)

    s = teaching_slide(
        'Technical detail: connect graphs to a language model',
        'Soft prompts are vectors. Numbered evidence text is also supplied for verification after generation.',
        'Components / Technical 2')
    eq(s, r'\mathbf{g}=\sum_i a_i\mathbf{h}_i', 151, h=47)
    label(s, 'h_i: subgraph; a_i: pooling weight; g: global vector', 55, 208, 502, size=16)
    arrow(s, (307, 244), (307, 258))
    card(s, 81, 265, 452, 73, 'Shared projector: 256 → 2048 → 4096', 'Map graph vectors into language-model inputs')
    arrow(s, (307, 341), (307, 355))
    card(s, 52, 365, 309, 63, '1 global + up to 8 local prompts')
    label(s, '＋', 363, 382, 25, TEAL, 21)
    card(s, 393, 365, 175, 63, 'Numbered evidence', color=GOLD, fill=GOLD_PALE)
    words(s, [
        ('What is trained?', 'Train graph matching first. During prompting, freeze the graph encoder and update pooling and the projector between representations.'),
        ('What does the LLM do?', 'Frozen Qwen3-8B selects components and citations. The program then checks valid identifiers, conditions, parameters, and units.'),
    ], 'The LLM organizes supported recommendations; evidence strength must still be judged for the specific use.')
    pages.append(s)
    assert len(pages) == 2
    return pages
