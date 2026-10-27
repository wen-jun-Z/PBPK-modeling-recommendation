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


def page(title, step, subtitle='Hypothetical drug X throughout · Values illustrate the method, not new experimental results'):
    return teaching_slide(title, subtitle, 'Blueprint / '+step)


def row(s, y, left, right, *, color=TEAL):
    """A compact, readable glossary or input/output pair."""
    s.rect(50, y, 522, 46, PALE if color == TEAL else GREEN_PALE, None)
    s.text(63, y+9, 185, left, st.fit_size(left,185,29,18,line=1.15,minimum=12), color, bold=True, line=1.15)
    h = s.text(252, y+9, 304, right, st.fit_size(right,304,29,17,line=1.15,minimum=12), INK, line=1.15)
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
    s = page('Define the task: what is known, and what will be predicted?', '01')
    card(s, 50, 151, 238, 124, 'Known: development data', 'Drug X, 70 kg adult\n100 mg infused over 1 hour\nA few time–concentration points')
    card(s, 331, 151, 241, 124, 'Target: new conditions', 'Same drug X and human\n50 mg infused over 1 hour\nFull curve and endpoints', GREEN, GREEN_PALE)
    arrow(s, (291, 213), (327, 213))
    card(s, 93, 307, 435, 104, 'Missing: an executable model', 'Measurements are data; a model is a set of equations\nthat maps dosing conditions to a full curve.', GOLD, GOLD_PALE)
    words(s, [('Why a blueprint?', 'No suitable model was found, but drug data and a few measurements may be available. Use them to build a model.'),
              ('What remains hidden?', 'Set target-condition measurements aside. Compare them only after prediction to evaluate predictive accuracy.')],
          'A few measurements are not a model; the goal is to turn available data into executable equations.')
    slides.append(s)

    # 2. Plain-language glossary; no mysterious abbreviations at first contact.
    s = page('Inputs: drug, human, dosing, and development observations', '02')
    row(s, 149, 'Drug identity', 'Name, molecular weight, sources')
    row(s, 202, 'Drug properties', 'Unbound fraction, lipophilicity, acidity')
    row(s, 255, 'Human and dosing', 'Age, weight; dose, infusion duration')
    row(s, 308, 'Development observations', 'Sampling times and concentrations')
    example(s, 'MW: mass per mole; fu: fraction not protein-bound; logP/pKa: lipophilicity/acidity.', y=365, h=67)
    words(s, [('Where do inputs come from?', 'Human data come from study conditions; drug properties from qualified measurements or literature. Retain units, sources, and measurement status.'),
              ('What is checked first?', 'Check drug identity and units, then harmonize time and concentration. Flag missing critical data instead of silently assuming them known.')],
          'Organize four input groups and trace every value to its source before computation.')
    slides.append(s)

    # 3. Fixed anatomy, with one organ enlarged rather than an unreadable network.
    s = page('Step 1: build a drug transport network from physiology', '03')
    s.asset('01_human_core.png', 50, 152, 127, 241)
    arrow(s, (182, 258), (230, 258))
    card(s, 239, 151, 327, 72, 'Blood flow connects organs', 'Template: volumes, flow, connections')
    card(s, 247, 253, 137, 96, 'Vascular', 'Drug enters and\nleaves with flow')
    card(s, 420, 253, 137, 96, 'Tissue', 'Drug is taken up\nor released', GREEN, GREEN_PALE)
    arrow(s, (388, 301), (416, 301), both=True)
    label(s, 'One organ enlarged: vascular ↔ tissue', 220, 364, 358)
    words(s, [('The human template', 'Fix organ volumes, flows, and connections. Fifteen organs each have vascular and tissue regions, connected by three blood pools.'),
              ('From drug data to parameters', 'Partition formulas map lipophilicity and unbound fraction to initial tissue capacities. Development data later calibrate unknown clearance and other terms.')],
          'Physiology defines the transport network; drug parameters determine movement into and out of organs.')
    slides.append(s)

    # 4. The four effective degrees of freedom and protected evidence.
    s = page('Step 2: preserve known inputs; allow bounded adjustments', '04')
    card(s, 50, 150, 522, 73, 'Protect reliably sourced inputs', 'Anatomy; qualified net renal CL and measured blood partitioning')
    for x, y, title, body in [(50,246,'Clearance','How fast drug disappears'),(318,246,'Tissue capacity','How much tissue can hold'),
                               (50,340,'Regional exchange','How fast drug enters tissue'),(318,340,'RBC partitioning','How much enters blood cells')]:
        card(s,x,y,254,77,title,body,GREEN,GREEN_PALE)
    words(s, [('Up to four adjustment axes', 'Four effective parameters can change only within allowed bounds. They do not permit arbitrary changes to anatomy.'),
              ('When measurements exist', 'Qualified partition measurements are protected, even if fitting is poor. Unmeasured parameters open only under defined rules.')],
          'Protect reliable known inputs and calibrate only the unknowns that are allowed to change.')
    slides.append(s)

    # 5. CL is a proportionality coefficient, not blood physically removed.
    s = page('Clearance: higher concentration means more elimination per hour', '05')
    card(s, 53, 153, 201, 89, 'Plasma concentration', '1 mg/L')
    card(s, 368, 153, 201, 89, 'Clearance CL', '9 L/h', GREEN, GREEN_PALE)
    arrow(s,(258,198),(365,198))
    label(s, 'Multiply them', 251, 166, 118)
    eq(s, r'\mathrm{Elimination\ rate}=CL\times C_p', 261, 55)
    label(s, 'Elimination rate (mg/h) = clearance × plasma concentration', 61, 322, 502)
    example(s, 'Drug X: 9 L/h × 1 mg/L = 9 mg eliminated per hour. Clearance does not mean physically removing 9 L of blood.')
    words(s, [('What does CL mean?', 'Clearance has units of volume/time. Multiplying it by concentration gives the actual amount eliminated per hour.'),
              ('If renal evidence exists', 'Keep qualified net renal clearance fixed and fit the remainder. This systemic approximation cannot identify specific metabolic enzymes.')],
          'Clearance is a proportionality factor; multiply it by concentration to calculate elimination per hour.')
    slides.append(s)

    # 6. Explain amount/capacity and difference before a general equation.
    s = page('Exchange follows a concentration difference', '06')
    card(s,50,151,220,115,'Vascular','4 mg in 2 L capacity\nEffective concentration = 2 mg/L')
    card(s,351,151,220,115,'Tissue','6 mg in 6 L capacity\nEffective concentration = 1 mg/L',GREEN,GREEN_PALE)
    arrow(s,(274,205),(347,205),both=True)
    label(s,'Two-way exchange',270,174,81,size=14)
    eq(s,r'J=5\times(2-1)=5\ \mathrm{mg/h}',282,51)
    label(s,'Exchange 5 L/h × concentration difference 1 mg/L',59,340,506)
    example(s,'At this hypothetical instant, tissue gains 5 mg/h and vessels lose 5 mg/h. Exchange does not create drug.')
    words(s,[('What does capacity mean?','Divide amount by equivalent capacity to obtain the concentration driving exchange. Capacity reflects partitioning, not anatomical volume.'),
             ('What sets the direction?','Drug enters tissue when vascular concentration is higher, and returns when tissue concentration is higher. Here blood/plasma ratio = 1.')],
          'Convert amounts to concentrations, then multiply their difference by the exchange coefficient.')
    slides.append(s)

    # 7. An actual differential equation anchored to physical arrows.
    s = page('Differential equations track change = inflow − outflow', '07')
    card(s,188,177,233,87,'An organ vascular region','Track its current drug amount')
    arrow(s,(54,217),(184,217)); label(s,'Blood-flow inflow',49,180,130)
    arrow(s,(425,217),(568,217)); label(s,'Blood-flow outflow',426,180,142)
    arrow(s,(303,266),(303,301),GREEN); label(s,'Net tissue influx J',328,273,185,color=GREEN)
    eq(s,r'\frac{dA_v}{dt}=Q(C_{in}-C_{out})-J',313,58)
    example(s,'With Q = 10 L/h, inflow/outflow concentrations 3/2 mg/L, and J = 5 mg/h, vascular amount rises by 5 mg/h.',y=374,h=58)
    words(s,[('Reading the symbols','Av is vascular drug amount; the left side is its hourly change. Q is flow, C is blood concentration, and J is net transfer into tissue.'),
             ('Why does this scale up?','Each organ follows this balance rule. Link organ flows and add dosing and clearance to form whole-body equations.')],
          'Differential equations track how much drug each region gains and loses per hour.')
    slides.append(s)

    # 8. Calculation input-output; matrix integral stays in appendix.
    s = page('Solve all organs together to obtain a complete curve', '08')
    card(s,50,151,148,96,'Parameters','CL / capacity\nexchange /\npartitioning')
    card(s,233,151,148,96,'Solve jointly','Calculate all\norgans together')
    card(s,417,151,155,96,'Read plasma','Concentration at\nevery time point',GREEN,GREEN_PALE)
    arrow(s,(201,193),(229,193)); arrow(s,(384,193),(413,193))
    st.curve(s,93,282,438,104,labels=False)
    label(s,'Vertical: plasma concentration',73,253,191,color=TEAL)
    label(s,'Horizontal: time after dosing',367,394,192)
    words(s,[('What does a solve need?','A fixed human network, candidate drug parameters, and dosing rate. Infusing 100 mg over 1 h gives an input of 100 mg/h.'),
             ('What is calculated?','Solve drug amounts over time, then convert to plasma concentration. Every new candidate parameter set requires a fresh curve calculation.')],
          'Given physiology, parameters, and dosing, the equation solver produces a predicted concentration curve.')
    slides.append(s)

    # 9. Optimization loop with observed dots, readable meaning and one loss.
    s = page('Step 3: calibrate unknowns to development observations', '09')
    card(s,50,150,151,79,'Try parameters','Within allowed bounds')
    card(s,235,150,151,79,'Solve curve','Compare dev. points')
    card(s,420,150,151,79,'Update','Keep better parameters',GREEN,GREEN_PALE)
    arrow(s,(204,190),(231,190)); arrow(s,(389,190),(416,190))
    s.line([(496,232),(496,253),(126,253),(126,233)],GOLD,1.6,dash=True,arrow=True)
    st.curve(s,85,279,448,102,dots=True,initial=True,labels=False)
    label(s,'Dots: development observations; line: candidate prediction',70,396,485)
    words(s,[('This is calibration','Try, solve, compare, and update repeatedly. The implementation uses bounded multistart least squares, not neural-network training.'),
             ('How is fitting constrained?','Use development points only, bound parameters, and penalize excessive partition changes. Select between two formulas using development error only.')],
          'Development points tune parameters; target-study observations must remain outside this loop.')
    slides.append(s)

    # 10. Feasibility diagnostic, rather than a generic fit-quality rescue.
    s = page('Step 4: check whether the allowed range is feasible', '10')
    card(s,51,151,521,84,'Sometimes the bounds make a fit impossible','A finite dose imposes an upper bound on visible concentration.')
    card(s,51,261,249,100,'Adequate range','Continue normal calibration;\ndo not loosen bounds arbitrarily')
    card(s,325,261,247,100,'Proven inadequate range','Only an unmeasured RBC lower bound\nmay be relaxed under the rules',GOLD,GOLD_PALE)
    example(s,'If the current maximum is 2 mg/L but development data reach 3 mg/L, first check sources, then repair eligibility.',y=375,h=57)
    words(s,[('How is impossibility shown?','Use administered amount and the concentration mapping to calculate a reachable upper bound. Only a clear exceedance triggers the check.'),
             ('Repair is not cosmetic','Measured partition values remain protected. Only eligible unmeasured bounds expand, followed by recalibration; accuracy gains are not guaranteed.')],
          'Assess feasibility before allowing repair; large error alone is not permission to change bounds.')
    slides.append(s)

    # 11. Clear lock and target branch, without hidden reference fitting.
    s = page('Step 5: lock the model, then predict new dosing', '11')
    st.lock(s,64,162,.78)
    card(s,128,151,441,85,'Lock selected parameters and partition formula','Record versions, sources, bounds, and repair reasons')
    card(s,50,270,224,95,'Enter target dosing','Drug X: 50 mg\nStill infused over 1 hour')
    card(s,347,270,224,95,'Predict target curve','Calculate without refitting\nto target concentrations',GREEN,GREEN_PALE)
    arrow(s,(278,316),(343,316))
    example(s,'With physiology, duration, and parameters fixed, this linear model gives half the curve at 50 versus 100 mg. Do not assume this for nonlinear drugs.',y=376,h=57)
    words(s,[('What may change after locking?','This example changes dose only; drug parameters stay fixed. A new population requires rebuilt physiology, not reuse of the full matrix.'),
             ('When are targets used?','Save predictions before reading target measurements for scoring. This evaluates prediction rather than fitting again.')],
          'Locking ends fitting. New inputs enter the solver; target answers are reserved for final scoring.')
    slides.append(s)

    # 12. First five outputs, curve and scalar distinction.
    s = page('Output 1: read four common endpoints from one curve', '12')
    st.curve(s,92,170,438,124,fill=True,labels=False)
    label(s,'C',51,145,65,color=TEAL)
    label(s,'Time',501,303,62)
    s.text(190,152,230,'Highest point → Cmax',16,TEAL)
    s.text(192,306,329,'Peak time → Tmax',16,TEAL)
    s.text(358,236,197,'Last time → Clast',16,GREEN)
    s.text(72,341,489,'Area under the curve → exposure over a time window',18,GREEN,bold=True)
    example(s,'From 1–2 h, concentration falls from 2 to 1 mg/L: trapezoid area = (2+1)÷2×1 = 1.5 mg·h/L.',y=376,h=56)
    words(s,[('Why can one curve give many outputs?','The curve supplies concentrations over time, giving the peak value, peak time, last concentration, and integrated area.'),
             ('Match the observations','Compare predictions at observed times and within the same window. The largest sparse sample may miss the actual continuous peak.')],
          'The first group has five output types: the curve, AUC, Cmax, Tmax, and last concentration.')
    slides.append(s)

    # 13. Six terminal quantities in a plain-language grid, no equation wall.
    s = page('Output 2: calculate six more endpoints if the tail is reliable', '13')
    boxes=[('Half-life','Time for terminal concentration to halve'),('AUC∞','Total exposure including the unobserved tail'),('Clearance CL','Dose divided by total exposure'),
           ('Vz','Distribution volume from terminal decline'),('MRT','Mean time drug remains in the body'),('Vss','Apparent steady-state distribution volume')]
    for i,(title,body) in enumerate(boxes):
        card(s,50+(i%2)*267,150+(i//2)*87,255,82,title,body,TEAL if i<3 else GREEN,PALE if i<3 else GREEN_PALE)
    label(s,'Vz and Vss are apparent volumes, not anatomical organ volumes.',52,431-19,519,size=15)
    words(s,[('Why check the tail first?','A short, flat, or unstable tail can distort half-life and derived endpoints. Mark them not estimable when checks fail.'),
             ('Are these copied inputs?','All endpoints come from the predicted trajectory. Record construction CL separately from trajectory-derived CL, then compare with references.')],
          'Together there are 11 output types. Explain non-estimable results instead of filling in unsupported numbers.')
    slides.append(s)

    # 14. A low-jargon explanation of scoring and its limits.
    s = page('Evaluate with observations excluded from calibration', '14')
    card(s,50,151,242,96,'Example 1','Predicted 3, observed 2\n1.5-fold error')
    card(s,330,151,242,96,'Example 2','Predicted 1, observed 2\n2-fold error',GREEN,GREEN_PALE)
    eq(s,r'\mathrm{GMFE}=\sqrt{1.5\times2}\approx1.73',262,55)
    label(s,'GMFE: geometric mean fold error; closer to 1 is better.',56,333,511,size=17)
    example(s,'These numbers teach scoring only. Real evaluation summarizes drugs, endpoints, failures, and uncertainty.')
    words(s,[('Interpreting fold error','Twice the observed value and half the observed value both count as a 2-fold error. Perfect agreement is 1, not 0.'),
             ('One number is insufficient','Score curves and endpoints separately. Examine worsening, unreliable tails, and source differences rather than inventing one universal score.')],
          'Development points fit parameters; target points score predictions. Fold errors near 1 indicate agreement.')
    slides.append(s)

    # 15. Make 4+1 optional and prevent the main-experiment misunderstanding.
    s = page('“4 + 1” tests extra observations; it does not define the blueprint', '15')
    card(s,50,151,227,96,'Initial construction','4 development points\n→ calibrate → predict')
    card(s,344,151,227,96,'Add one development point','Choose by time coverage\n→ recalibrate → predict',GREEN,GREEN_PALE)
    arrow(s,(281,193),(340,193))
    card(s,90,285,442,112,'Does one extra observation add practical value?','Compare prediction errors on the same target study.\nTarget measurements remain scoring-only.',GOLD,GOLD_PALE)
    words(s,[('Can all five points be used?','Yes. Calibrate directly with five available points, or more qualified development points. There is no need to hide one artificially.'),
             ('How is the point chosen?','Use a simple time-coverage rule. Current results do not show universal gains from a complex learned selector, so it is not required.')],
          'The main method constructs and predicts; 4 + 1 only measures the benefit of another development observation.')
    slides.append(s)

    # 16. A defensible contribution, not stronger claims than evidence supports.
    s = page('Our contribution: automated, constrained, traceable construction', '16')
    card(s,50,150,251,100,'Evidence: protect','Keep sources and units\nPreserve qualified measurements')
    card(s,321,150,251,100,'Unknowns: bound','Calibrate within allowed bounds\nRecord selection reasons',GREEN,GREEN_PALE)
    card(s,50,276,251,100,'Conflicts: diagnose','Check range feasibility\nChange only eligible terms',GOLD,GOLD_PALE)
    card(s,321,276,251,100,'Outputs: qualify','Predict after locking\nReport quality and unsupported uses')
    words(s,[('Where is the contribution?','Connect drug evidence, bounded calibration, feasibility checks, and prediction quality in a reproducible workflow, reusing existing physiology.'),
             ('Limits of the claims','Current scope is single-dose IV plasma prediction using drug-specific observations. Oral, organ, or unseen-drug accuracy and clinical safety are unproven.')],
          'Requests and evidence yield an executable model, predictions, construction rationale, and explicit limitations.')
    slides.append(s)
    assert len(slides) == 16
    return _check(slides)


def make_blueprint_technical_slides():
    """Five optional equation pages, after the beginner-first main narrative."""
    slides=[]

    s=page('Technical 1: map four adjustments to model parameters','Equation 1/5','Initial capacities: PK-Sim Standard / Rodgers–Rowland; calibrate each and select by development loss.')
    eq(s,r'CL=CL_R+s_{CL}(CL_0-CL_R)',151,48)
    eq(s,r'K_{ie}=s_KK_{ie,0},\quad PS_i=s_{ex}Q_{i,eff}',218,48)
    eq(s,r'K_{RBC}=s_{RBC}K_{RBC,0}',285,45)
    eq(s,r'R_{BP}=1-Hct+Hct\,K_{RBC}',348,45)
    label(s,'0: baseline; s: allowed multiplier; i: organ index.',55,412,514,size=15)
    words(s,[('Clearance and tissue axes','CL is total clearance; CL_R is fixed net renal clearance. Kie is tissue capacity; PS is exchange scaled by effective flow Q.'),
             ('The RBC axis','K_RBC is the blood-cell/plasma partition ratio; Hct is hematocrit. R_BP is the blood/plasma ratio and updates with RBC partitioning.')],
          'These mappings turn a few adjustable values into the parameters required by each equation solve.')
    slides.append(s)

    s=page('Technical 2: organ exchange and amount equations','Equation 2/5','Illustrative ordinary organ; the full network retains multiple inlets such as portal flow')
    eq(s,r'K_{iv}=V_{i,pl}+K_{RBC}V_{i,RBC}',151,46)
    eq(s,r'J_i=PS_iR_{BP}\left(\frac{A_{iv}}{K_{iv}}-\frac{A_{ie}}{K_{ie}}\right)',213,64)
    eq(s,r'\frac{dA_{iv}}{dt}=Q_i(C_{a,b}-C_{iv,b})-J_i',286,59)
    eq(s,r'\frac{dA_{ie}}{dt}=J_i',366,53)
    words(s,[('Vascular and tissue regions','A is amount; v/e denote vascular/tissue regions. K is equivalent capacity, V anatomical volume, and pl/RBC plasma/blood cells.'),
             ('Concentrations and flows','Ca,b is incoming arterial concentration; Civ,b is outgoing blood concentration. Q is flow; equal and opposite J terms conserve exchanged drug.')],
          'Tissue exchange moves drug; dosing and clearance change the whole-body amount.')
    slides.append(s)

    s=page('Technical 3: the whole-body linear system and plasma curve','Equation 3/5','15 organs × 2 regions + 3 blood pools = 33 body amount states; elimination is tracked separately')
    eq(s,r'\dot{\mathbf{A}}=\mathbf{M}\mathbf{A}+\mathbf{b}\,r(t)',147,52)
    eq(s,r'r(t)=D/T\ (0\leq t\leq T),\quad 0\ (t>T)',210,43)
    eq(s,r'\mathbf{A}(t)=\int_0^{\min(t,T)} e^{\mathbf{M}(t-\tau)}\mathbf{b}\frac{D}{T}\,d\tau',268,67)
    eq(s,r'\widehat C_p(t)=\mathbf{h}^{\mathsf{T}}\mathbf{A}(t)',354,49)
    words(s,[('Matrix and inputs','A contains drug amounts; M combines flow, capacity, exchange, and clearance. b selects the dosing inlet; D is dose and T infusion duration.'),
             ('Reading concentration','Start with zero drug and calculate matrix exponentials. h converts blood amounts to sampled plasma concentration and updates with partitioning.')],
          'This linear system is derived from physiology; it does not reproduce every native PK-Sim mechanism.')
    slides.append(s)

    s=page('Technical 4: calibration objective and feasibility bound','Equation 4/5','Actual algorithm: bounded multistart least squares, not a blueprint neural network')
    eq(s,r'L=\frac{1}{n}\sum_j\left[\ln\frac{\widehat C_p(t_j)}{C_j}\right]^2+0.01R',146,67)
    eq(s,r'R=\left(\frac{\ln s_K}{\ln10}\right)^2+\left(\frac{\ln s_{ex}}{\ln100}\right)^2',225,61)
    eq(s,r'\qquad+\left(\frac{\ln s_{RBC}}{\ln10}\right)^2',294,61)
    eq(s,r'C_p(t)\leq D_{delivered}(t)\max_i h_i',369,47)
    words(s,[('Development loss','Cj is observation j at time tj; n is the number of points. Compare predictions at the same times. R penalizes large partition changes, not CL.'),
             ('A feasibility bound only','Administered amount times the largest concentration coefficient gives a necessary bound. Repair requires exceedance and unmeasured partitioning.')],
          'A better fit does not imply unique parameters; equally good fits may extrapolate differently.')
    slides.append(s)

    s=page('Technical 5: deriving terminal endpoints from the curve','Equation 5/5','Single-dose IV/infusion context; calculate only after quality checks and leave unsupported outputs missing')
    eq(s,r't_{1/2}=\frac{\ln2}{\lambda_z},\quad AUC_\infty=AUC_{0-t}+\frac{C_{last}}{\lambda_z}',146,69)
    eq(s,r'CL_{NCA}=\frac{D}{AUC_\infty},\quad V_z=\frac{CL_{NCA}}{\lambda_z}',234,61)
    eq(s,r'MRT=\frac{AUMC_\infty}{AUC_\infty}-\frac{T}{2}',313,53)
    eq(s,r'V_{ss}=CL_{NCA}\,MRT',382,43)
    words(s,[('Tail and areas','λz is the negative terminal log-concentration slope. AUMC is time-weighted concentration area; T is infusion duration and D dose.'),
             ('Separate inputs and outputs','CL_NCA is trajectory-derived clearance; Vz and Vss are apparent volumes. Failed terminal regression or extrapolation blocks derived endpoints.')],
          'Check tail quality before calculating derived endpoints; an unstable tail cannot yield a reliable half-life.')
    slides.append(s)
    assert len(slides)==5
    return _check(slides)


if __name__ == '__main__':
    main=make_blueprint_chain_slides()
    technical=make_blueprint_technical_slides()
    print({'beginner_blueprint':len(main),'technical_backup':len(technical),'speaker_notes':'empty'})
