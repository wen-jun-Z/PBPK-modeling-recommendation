#!/usr/bin/env python3
"""Explain the frozen blueprint implementation and two auditable worked cases.

No model is fitted, no experiment or paper is overwritten. Re-evaluate frozen
parameters to show intermediate calculations; verify them against saved fits.
Final outputs: one editable deck, matching PDF and contact preview.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np

from build_blueprint_method_slides import (
    ROOT, OUT, Slide, render, TEAL, INK, MUTED, GREEN, GOLD, PURPLE,
    PALE, GREEN_PALE, GOLD_PALE, PURPLE_PALE, WHITE, LINE, FONT_FILE,
    EQUATION_FONT_SIZE, fitz, Image, fit_size,
)

sys.path.insert(0, str(ROOT / 'src'))
from pbpk_kb.blueprint_policy import blood_partition_calibration as blood
from pbpk_kb.blueprint_policy import permeability_model as exchange
from pbpk_kb.blueprint_policy.perfusion_calibration import clearance_inputs

STEM = 'PBPK_blueprint_from_physiology_to_prediction'
PINS = {}


def read(path):
    if isinstance(path, dict):
        ref = path
        path = ref['path']
    else:
        ref = {}
    p = ROOT / path
    raw = p.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if ref.get('sha256'):
        assert sha == ref['sha256'], f'Source changed: {path}'
    PINS[str(p.relative_to(ROOT))] = sha
    return json.loads(raw)


def cases():
    manifest = read('results/tables/blueprint_sampling_clinical_sources.json')
    data = {}
    for row in manifest['cases']:
        if row['drug'] not in {'Cefepime', 'Montelukast'}:
            continue
        result = read(row['time_repaired']['result'])
        training = read(result['training'])
        formula = result['formula']
        if isinstance(result['fits'], list):
            fits = {r['formula']: read(r['fit']) for r in result['fits']}
            source_fit = fits[formula]
        else:
            fits = {r['formula']: r['fit'] for r in read(result['fits'])}
            original = read(result['original_source'])
            source_fit = read(next(r['fit'] for r in original['fits'] if r['formula'] == formula))
        request = source_fit['request']
        phys = read(source_fit['development_physiology'])
        logs = np.array(result['parameters']['log_parameters'])
        times = np.array([r['time_min'] / 60 for r in training['observations']])
        observed = np.array([r['concentration_umol_per_l'] for r in training['observations']]) * phys['molecular_weight_g_mol'] / 1000
        predicted = blood.simulate(phys, request, times, logs)['concentration_mg_l']
        log_errors = np.log(predicted / observed)
        mse = float(np.mean(log_errors**2))
        penalty = .01*((logs[1]/math.log(10))**2+(logs[2]/math.log(100))**2+(logs[3]/math.log(10))**2)
        gmfe = math.exp(float(np.mean(np.abs(log_errors))))
        assert np.isclose(mse + penalty, result['parameters']['objective'], rtol=1e-7)
        assert np.isclose(gmfe, result['parameters']['development_gmfe'], rtol=1e-7)
        grid = np.unique(np.r_[np.linspace(.001, times[-1], 700), times, request['dosing']['infusion_minutes']/60])
        fitted_curve = blood.simulate(phys, request, grid, logs)['concentration_mg_l']
        prediction = read(row['time_repaired']['prediction'])
        reference = read(row['target_reference'])
        if 'cases' in reference:
            rc = next(c for c in reference['cases'] if c['case_id'] == row['case_id'])['clinical_reference']['curve']
            target_times = np.array(rc['time_h'])
            assert rc['unit'] == 'ng/mL'
            target_obs = np.array(rc['concentration'])/1000
        else:
            target_times = np.array([v['time_min']/60 for v in reference['observations']])
            target_obs = np.array([v['concentration_umol_per_l'] for v in reference['observations']])*phys['molecular_weight_g_mol']/1000
        target_grid = np.array(prediction['time_min'])/60
        target_curve = np.array(prediction['concentration'])*phys['molecular_weight_g_mol']/1000
        target_pred = np.interp(target_times, target_grid, target_curve)
        target_gmfe = math.exp(float(np.mean(np.abs(np.log(target_pred/target_obs)))))
        assert np.isclose(target_gmfe, row['target_metrics']['time_repaired']['gmfe'], rtol=1e-6)
        derived = blood.derive_physiology(phys, request, logs[3])
        cl = clearance_inputs(request)
        system = exchange.build_system(derived, total_clearance_l_h=cl['total_clearance_l_h'],
            renal_clearance_l_h=cl['renal_clearance_l_h'], log_parameters=dict(zip(blood.PARAMETERS[:3], logs[:3])))
        data[row['drug']] = dict(row=row,result=result,training=training,fit=source_fit,fits=fits,
            request=request,target_request=read(row['time_repaired']['request']),phys=phys,derived=derived,
            logs=logs,cl=cl,system=system,times=times,observed=observed,predicted=predicted,
            mse=mse,penalty=penalty,gmfe=gmfe,grid=grid,fitted_curve=fitted_curve,
            target_times=target_times,target_obs=target_obs,target_pred=target_pred,
            target_grid=target_grid,target_curve=target_curve,target_gmfe=target_gmfe)
        print(row['drug'], {'times_h':times.tolist(),'observed_mg_l':observed.tolist(),
            'fitted_mg_l':predicted.tolist(),'mse':mse,'penalty':penalty,'gmfe':gmfe,
            'target_gmfe':target_gmfe,'CL':system.metadata['effective_total_clearance_l_h'],
            'RBC':derived['rbc_kp_plasma'],'BP':derived['blood_to_plasma_ratio']})
    return data


def charts(data, work):
    """Clinical observations are points; simulations are lines, no invented CI."""
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import LogLocator, NullFormatter
    skill = Path('/home/xuyiqian/.codex/skills/scipilot-figure-skill/scripts')
    sys.path.insert(0,str(skill))
    from profile_data import profile_data
    from visual_qa import audit_layout
    records=[]
    for drug,c in data.items():
        for group,ts,ys in [('development',c['times'],c['observed']),('target',c['target_times'],c['target_obs'])]:
            records.extend(dict(drug=drug,group=group,time_h=float(t),concentration_mg_l=float(y)) for t,y in zip(ts,ys))
    profile=profile_data(pd.DataFrame(records),group_cols=['drug','group'])
    assert all(np.isfinite(r['concentration_mg_l']) and r['concentration_mg_l']>0 for r in records)
    print('Clinical plotting profile:',len(records),'points;',[(k,len(v['times']),len(v['target_times'])) for k,v in data.items()])
    paths={}
    for drug,c in data.items():
        fig,axes=plt.subplots(1,2,figsize=(876/72,274/72),layout='constrained')
        for ax,group,color,marker in zip(axes,['Development','Target'],['#0072B2','#D55E00'],['o','s']):
            dev=group=='Development'
            ts,ys=(c['times'],c['observed']) if dev else (c['target_times'],c['target_obs'])
            gx,gy=(c['grid'],c['fitted_curve']) if dev else (c['target_grid'],c['target_curve'])
            mask=(gx>0)&(gx<=ts[-1])&(gy>0)
            ax.plot(gx[mask],gy[mask],color=color,lw=1.8,ls='-' if dev else '--')
            ax.scatter(ts,ys,s=35,marker=marker,facecolor='white',edgecolor=color,lw=1.5,zorder=5)
            ax.set_yscale('log');ax.set_xlim(0,ts[-1]*1.025)
            ax.set_ylim(min(ys)*.4,max(max(ys),max(gy[mask]))*1.4)
            ax.set_xlabel('Time after dose (h)',fontsize=13)
            ax.set_ylabel('Plasma concentration (mg/L)',fontsize=13)
            ax.tick_params(labelsize=12);ax.yaxis.set_major_locator(LogLocator(base=10,numticks=5))
            ax.yaxis.set_minor_formatter(NullFormatter());ax.grid(True,which='major',lw=.6,color='#E3E9ED')
            for sp in ('top','right'):ax.spines[sp].set_visible(False)
            for sp in ('left','bottom'):ax.spines[sp].set_color('#8899A6')
            labels=['Fit + 5 observations' if dev else f'Prediction + {len(ts)} observations']
            handle=Line2D([],[],color=color,lw=1.8,ls='-' if dev else '--',marker=marker,mfc='white',mec=color)
            ax.legend([handle],labels,loc='lower left',bbox_to_anchor=(0,1.01),frameon=False,fontsize=12,borderaxespad=0)
        fig.canvas.draw()
        issues=audit_layout(fig)
        assert not any(level=='FAIL' for level,_ in issues),issues
        print('Plot layout:',drug,issues)
        png=work/f'{drug}.png';pdf=work/f'{drug}.pdf'
        fig.savefig(png,dpi=220);fig.savefig(pdf)
        with Image.open(png) as im:im.convert('L').save(work/f'{drug}_gray.png')
        plt.close(fig);paths[drug]=(png,pdf)
    return paths


def page(title, subtitle, stage):
    s=Slide(title,subtitle,stage,'')
    s.ops[0]['rows']=['PBPK BLUEPRINT  /  '+stage]
    s.ops[4]['rows']=['Human template → parameter mapping → observational calibration → feasibility checks → locked prediction']
    return s


def txt(s,x,y,w,t,size=17,color=INK,bold=False,align='left'):
    return s.text(x,y,w,t,size,color,bold=bold,align=align,line=1.22)


def box(s,x,y,w,h,title,body='',color=TEAL,fill=PALE,size=19):
    s.rect(x,y,w,h,fill,color,.8)
    th=txt(s,x+12,y+11,w-24,title,size,color,True,'center')
    if body:
        bh=txt(s,x+12,y+th+14,w-24,body,16,INK,align='center')
        assert th+14+bh+4<=h,(title,th,bh,h)


def arr(s,points,color=TEAL,both=False,dash=False):
    s.line(points,color,1.7,dash=dash,arrow=True,both=both)


def eq(s,latex,y,h=58,x=50,w=530):
    latex=re.sub(r'\\(mathbf|mathsf|mathcal)\s+([A-Za-z])',lambda m:'\\'+m[1]+'{'+m[2]+'}',latex)
    s.eq(x,y,w,h,latex,EQUATION_FONT_SIZE,INK)


def side(s,a,b):
    s.line([(601,147),(601,431)],'DFE7EA',.8)
    for y,(title,body) in zip([156,292],[a,b]):
        txt(s,624,y,290,title,fit_size(title,290,30,20,line=1.22),TEAL,True)
        h=txt(s,624,y+36,290,body,fit_size(body,290,92,17,line=1.22))
        assert h<=92,(title,body)


def carry(s,t):
    s.rect(42,451,876,39,PALE,None)
    h=txt(s,56,459,848,t,fit_size(t,848,27,16,line=1.22,minimum=11),TEAL)
    assert h<28,t


def table(s,headers,rows,x=50,y=155,widths=None,row_h=39,size=17):
    widths=widths or [526/len(headers)]*len(headers)
    w=sum(widths)
    s.rect(x,y,w,row_h,TEAL,None,rounded=False)
    for j,row in enumerate([headers]+rows):
        yy=y+j*row_h
        if j:s.rect(x,yy,w,row_h,'F1F5F7' if j%2 else WHITE,None,rounded=False)
        xx=x
        for cell,cw in zip(row,widths):
            h=txt(s,xx+8,yy+9,cw-16,str(cell),size,WHITE if not j else INK,bold=not j,align='center')
            assert h<=row_h-8,(cell,row_h,h)
            xx+=cw


def pic(s,paths,x,y,w,h):
    png,pdf=paths
    s.ops.append(dict(kind='image',path=str(png),pdf=str(pdf),x=x,y=y,w=w,h=h))


def human_artwork(work):
    """Reuse the authored anatomy icon: native paths in PDF, clear PNG in PPTX."""
    import xml.etree.ElementTree as ET
    import cairosvg
    from blueprint_native_anatomy import native_anatomy
    ns='http://www.w3.org/2000/svg'
    root=ET.Element('{'+ns+'}svg',width='160',height='220',viewBox='35 80 160 220')
    root.append(native_anatomy('human'))
    svg=ET.tostring(root)
    pdf=work/'human.pdf';png=work/'human.png'
    cairosvg.svg2pdf(bytestring=svg,write_to=str(pdf))
    cairosvg.svg2png(bytestring=svg,write_to=str(png),output_width=640,output_height=880)
    return png,pdf


def slides(data, plots):
    """Eight-slide explanation: one message and a few essential equations per page."""
    from build_blueprint_method_slides import lock, curve
    pages=[]; c=data['Cefepime']
    def add(title,sub,stage):
        slide=page(title,sub,stage);pages.append(slide)
        slide.ops[4]['rows']=['Inputs and construction → observational calibration → constraint checks → locked prediction']
        return slide
    def label(slide,x,y,w,value,color=TEAL,size=18):
        return txt(slide,x,y,w,value,fit_size(value,w,31,size,line=1.22,minimum=12),color,True,'center')
    def card(slide,x,y,w,h,title,body='',color=TEAL,fill=PALE):
        slide.rect(x,y,w,h,fill,color,.8)
        label(slide,x+10,y+11,w-20,title,color,19)
        if body:txt(slide,x+12,y+46,w-24,body,fit_size(body,w-24,h-51,17,line=1.22,minimum=12),INK,align='center')
    def math(slide,latex,x,y,w,h=62,color=INK):
        slide.eq(x,y,w,h,latex,EQUATION_FONT_SIZE,color)

    # 1 — Establish the complete task before showing the calculation details.
    s=add('Blueprint design: evidence becomes a testable model',
          'When no suitable existing model is available, combine a human template, drug data, and a few development observations.','One workflow')
    for i,(title,body,col,fill) in enumerate([
        ('Inputs','Human + drug data\nDose + observations',TEAL,PALE),
        ('Build equations','Blood flow links organs\nDistribution and clearance',GREEN,GREEN_PALE),
        ('Calibrate and check','Adjust allowed parameters\nCheck bound conflicts',GOLD,GOLD_PALE),
        ('Predict after locking','New dosing conditions\nCurve + endpoints',PURPLE,PURPLE_PALE)]):
        x=50+220*i;card(s,x,166,198,125,title,body,col,fill)
        if i<3:arr(s,[(x+202,228),(x+215,228)])
    label(s,58,324,846,'Method focus: what is fixed, what is adjustable, and when repair is allowed',TEAL,21)
    for x,title,col,fill in [(73,'Protect reliable evidence',TEAL,PALE),(365,'Constrain unknowns',GREEN,GREEN_PALE),
                             (657,'Diagnose and repair',GOLD,GOLD_PALE)]:
        s.rect(x,377,230,43,fill,None);label(s,x,385,230,title,col,18)
    carry(s,'Deliver a traceable model that can be validated repeatedly; the human template itself is established methodology.')

    # 2 — One capacity relation connects the input evidence to the state equations.
    s=add('Physiology supplies structure; drug data control distribution and clearance',
          'Physiology supplies volumes and flow. Partition formulas map drug data to model parameters.','Inputs and construction')
    pic(s,plots['human'],51,156,121,270)
    card(s,195,153,302,105,'Human template','Organ volumes, composition, flow\n15 organs × 2 regions + 3 pools',TEAL,PALE)
    card(s,195,284,302,123,'Drug and dosing data','fu, logP, pKa, molecular weight, CL\nDose, infusion duration, sources',GREEN,GREEN_PALE)
    arr(s,[(502,204),(531,204),(531,278),(555,278)])
    arr(s,[(502,345),(531,345),(531,278)],GREEN)
    s.rect(565,153,344,254,PALE,TEAL,.8)
    label(s,578,166,318,'Volume × partitioning → equivalent capacity',TEAL,19)
    math(s,r'K=\sum_c V_c k_c',584,212,305,66)
    math(s,r'C_{equiv}=A/K',584,293,305,58)
    txt(s,583,368,310,'A: drug amount; K: equivalent drug-holding capacity.',16,MUTED,align='center')
    carry(s,'Initial partitioning uses PK-Sim Standard or Rodgers–Rowland; K is not anatomical volume.')

    # 3 — The model's actual organ dynamics, not an optimizer derivation.
    s=add('Core equations: blood flow transports drug; vascular and tissue regions exchange it',
          'Each organ has vascular and tissue regions, linked by flow to form the current linear differential equations.','Physiological core equations')
    card(s,225,153,237,83,'Vascular: amount Av','Capacity Kv',TEAL,PALE)
    card(s,652,153,251,83,'Tissue: amount Ae','Capacity Ke',GREEN,GREEN_PALE)
    arr(s,[(58,193),(219,193)]);label(s,54,157,158,'Flow in',TEAL,17)
    arr(s,[(468,193),(646,193)],GREEN,both=True);label(s,477,156,160,'Exchange J',GREEN,18)
    arr(s,[(345,239),(345,261),(520,261)]);label(s,370,234,159,'Flow out',TEAL,16)
    math(s,r'J=s_{ex}Q_{eff}R_{BP}\left(\frac{A_v}{K_v}-\frac{A_e}{K_e}\right)',92,278,780,72)
    math(s,r'\frac{dA_v}{dt}=Q(C_{in,b}-C_{v,b})-J',67,360,525,68)
    math(s,r'\frac{dA_e}{dt}=J',621,360,267,68)
    carry(s,'Q is blood flow, RBP the blood/plasma ratio, and A/K effective concentration. J follows the concentration difference.')

    # 4 — Only the four effects and the actual development objective are needed.
    s=add('Fit four multipliers to development observations',
          'Anatomical volumes, flow, and sampling weights stay fixed. Unknowns vary only within allowed bounds.','Observational calibration')
    for i,(title,body,col,fill) in enumerate([
        ('Clearance sCL','Scale clearance',TEAL,PALE),('Tissue sK','Scale tissue capacity',GREEN,GREEN_PALE),
        ('Exchange sex','Scale exchange rate',GOLD,GOLD_PALE),('Blood cells sRBC','Scale RBC partitioning',PURPLE,PURPLE_PALE)]):
        card(s,50+i*220,150,198,89,title,body,col,fill)
    for x,t,col in [(75,'Try multipliers',TEAL),(370,'Solve the curve',GREEN),(665,'Compare observations',GOLD)]:
        s.rect(x,267,220,41,PALE,None);label(s,x,274,220,t,col,18)
        if x<665:arr(s,[(x+227,287),(x+287,287)])
    math(s,r'\hat{\mathbf{s}}=\arg\min_{\mathbf{s}\in\mathcal{S}}\left[\frac{1}{n}\sum_{j=1}^{n}\left(\ln\frac{\hat C_p(t_j;\mathbf{s})}{C_{obs}(t_j)}\right)^2+P(\mathbf{s})\right]',52,325,856,85)
    txt(s,82,414,797,'Objective = pointwise concentration error + adjustment penalty. S defines bounds; P penalizes large changes.',17,INK,align='center')
    carry(s,'Use development points only. Calibrate both partition formulas and select by development loss; a good fit does not ensure unique parameters.')

    # 5 — Keep the decision evidence, omit numerical optimizer mechanics.
    s=add('Protect evidence; repair only diagnosed conflicts',
          'Check whether permitted parameter ranges can explain visible observations before allowing a bound adjustment.','Constraints and repair')
    s.rect(50,151,418,282,PALE,TEAL,.8)
    label(s,66,164,386,'① Protect qualified inputs',TEAL,20)
    txt(s,73,210,372,'Example: keep net renal CL_R fixed;\ncalibrate only remaining clearance.',19,INK)
    math(s,r'CL=CL_R+s_{CL}(CL_0-CL_R)',62,285,394,62)
    txt(s,72,371,374,'Without renal-split evidence, CL_R = 0.',16,MUTED)
    s.rect(492,151,418,282,GOLD_PALE,GOLD,.8)
    label(s,508,164,386,'② Check a conservation upper bound',GOLD,20)
    math(s,r'C_p(t)\leq D_{delivered}(t)\max_i h_i',508,213,386,64)
    txt(s,513,292,375,'Montelukast: bound ≈0.029 mg/L;\nobserved development value 0.373 mg/L.',17,INK)
    txt(s,513,351,375,'No conflict: lock directly.\nConflict: repair only an eligible unmeasured RBC lower bound; otherwise stop.',17,GOLD)
    carry(s,'Qualified blood-partition measurements block free RBC calibration; reliable inputs are not changed merely to reduce error.')

    # 6 — Locked parameters still permit target-human and dosing changes.
    s=add('After locking, predict curves and endpoints under new inputs',
          'Lock the partition formula and four multipliers. Rebuild for new physiology; dose and duration enter the input term.','Locked prediction')
    for i,(title,body,col,fill) in enumerate([
        ('Lock','Partition formula + four multipliers',TEAL,PALE),('Change inputs','Human + dose + infusion duration',GOLD,GOLD_PALE),
        ('Predict','Plasma curve + endpoints',GREEN,GREEN_PALE)]):
        x=50+i*295;card(s,x,149,270,89,title,body,col,fill)
        if i<2:arr(s,[(x+275,192),(x+288,192)],col)
    math(s,r'\frac{d\mathbf{A}}{dt}=M(\hat{\mathbf{s}};\mathcal{H}_{target})\mathbf{A}+\mathbf{b}r_{target}(t)',50,273,555,73)
    math(s,r'\hat C_p=\mathbf{h}_{target}^{\mathsf{T}}\mathbf{A}',70,360,510,60)
    s.rect(632,264,278,164,GREEN_PALE,GREEN,.8)
    label(s,644,277,254,'One curve → 11 output types',GREEN,18)
    txt(s,649,318,243,'Curve / AUC / Cmax / Tmax\nClast / half-life / AUC∞\nCL / Vz / MRT / Vss',17,INK,align='center')
    carry(s,'M combines flow, exchange, and clearance; h maps amounts to sampled concentration. Target observations are used only for scoring.')

    # 7 — One real worked example replaces the step-by-step arithmetic pages.
    s=add('Cefepime: calibrate at 30 min, predict at 3 min',
          'Calibrate on five development concentrations, then freeze parameters. Target-study observations are excluded from refitting.','A complete example')
    label(s,54,147,425,'Development: 2000 mg / 30 min / 79 kg',TEAL,18)
    label(s,488,147,425,'Target: 2000 mg / 3 min / 70 kg¹',GOLD,18)
    pic(s,plots['Cefepime'],48,180,868,264)
    carry(s,f'Development GMFE {c["gmfe"]:.2f} → target {c["target_gmfe"]:.2f}; closer to 1 is better. ¹Target assumes 70 kg.')

    # 8 — Separate the existing physics from the project's construction policy.
    s=add('Our contribution: evidence-based construction and adjustment',
          'Reuse physiology and partition formulas; focus on evidence constraints, feasibility diagnosis, and traceable delivery.','Focus and deliverables')
    for x,title,body,col,fill in [
        (50,'Evidence protection','Record sources and units\nProtect qualified inputs',TEAL,PALE),
        (345,'Bounded calibration','Explicit roles for unknowns\nPrespecified multiplier bounds',GREEN,GREEN_PALE),
        (640,'Diagnose and repair','Check physical feasibility\nRepair only eligible bounds',GOLD,GOLD_PALE)]:
        card(s,x,166,270,145,title,body,col,fill)
    label(s,65,346,830,'Deliver: model and parameters + curves and endpoints + source, calibration, and repair records',TEAL,21)
    txt(s,74,395,810,'Check each endpoint for calculability; flag insufficient sources or inadequate quality.',18,INK,align='center')
    carry(s,'Current scope: single-dose IV plasma prediction with a few drug-specific development observations. This case does not establish all-drug performance.')
    assert len(pages)==8
    return pages


def main():
    suffixes=('.pptx','.pdf','_preview.png')
    before={suffix:hashlib.sha256((OUT/(STEM+suffix)).read_bytes()).hexdigest()
            for suffix in suffixes if (OUT/(STEM+suffix)).exists()}
    data=cases()
    work_root=ROOT/'runtime/presentation_build';work_root.mkdir(exist_ok=True,parents=True)
    with tempfile.TemporaryDirectory(prefix='blueprint_walkthrough-',dir=work_root) as tmp:
        work=Path(tmp);plots=charts({'Cefepime':data['Cefepime']},work)
        plots['human']=human_artwork(work)
        pages=slides(data,plots)
        for i,s in enumerate(pages,1):
            assert len(s.ops[1]['rows'])==1,s.ops[1]
            assert len(s.ops[2]['rows'])==1,s.ops[2]
            s.text(855,503,63,f'{i:02d} / {len(pages):02d}',10,TEAL,align='right')
            s.notes=''
        from matplotlib.mathtext import MathTextParser
        from matplotlib.font_manager import FontProperties
        parser=MathTextParser('path');errors=[]
        for i,s in enumerate(pages,1):
            for op in s.ops:
                if op['kind']!='equation':continue
                try:
                    dims=parser.parse('$'+op['latex']+'$',dpi=72,prop=FontProperties(size=24,math_fontfamily='stix'))
                    if dims.width>op['w'] or dims.height>op['h']:
                        errors.append((i,dims.width,dims.height,op['latex']))
                except ValueError as e:errors.append((i,str(e)))
        assert not errors,errors
        toc=[[1,item.ops[0]['rows'][0].split('/  ')[-1],i] for i,item in enumerate(pages,1)]
        render(pages,output_dir=work,stem=STEM,title='PBPK blueprint design: physiology to prediction (8-slide overview)',
            required_terms=['Human','Differential equations','Calibration','Cefepime','Montelukast','Concentration'],
            toc=toc)
        # Frozen scientific inputs must remain unchanged throughout the build.
        for path,sha in PINS.items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha,path
        # Keep full-size review pages only in the project runtime, not results.
        review=work_root/'blueprint_walkthrough_review';review.mkdir(exist_ok=True)
        for old_page in review.glob('page_*.png'):
            old_page.unlink()
        with fitz.open(work/(STEM+'.pdf')) as pdf:
            for i,p in enumerate(pdf):
                p.get_pixmap(matrix=fitz.Matrix(1.3,1.3)).save(review/f'page_{i+1:02d}.png')
        for suffix,sha in before.items():
            assert hashlib.sha256((OUT/(STEM+suffix)).read_bytes()).hexdigest()==sha, 'Presentation changed during build'
        for suffix in suffixes:
            (work/(STEM+suffix)).replace(OUT/(STEM+suffix))
        print({'pages':len(pages),'verified_source_files':len(PINS),'formula_pt':EQUATION_FONT_SIZE,'new_model_fits':0})


if __name__ == '__main__':
    main()
