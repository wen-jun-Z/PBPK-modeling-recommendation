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
    EQUATION_FONT_SIZE, fitz, Image,
)

sys.path.insert(0, str(ROOT / 'src'))
from pbpk_kb.blueprint_policy import blood_partition_calibration as blood
from pbpk_kb.blueprint_policy import permeability_model as exchange
from pbpk_kb.blueprint_policy.perfusion_calibration import clearance_inputs

STEM = 'PBPK蓝图设计_人体模板到预测_方法与实例'
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
    s.ops[4]['rows']=['人体模板 → 参数映射 → 观测校准 → 可行性检查 → 锁定预测']
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
        txt(s,624,y,290,title,20,TEAL,True)
        h=txt(s,624,y+36,290,body,17)
        assert h<=92,(title,body)


def carry(s,t):
    s.rect(42,451,876,39,PALE,None)
    h=txt(s,56,459,848,t,16,TEAL)
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
        slide.ops[4]['rows']=['输入与构建 → 观测校准 → 约束检查 → 锁定预测']
        return slide
    def label(slide,x,y,w,value,color=TEAL,size=18):
        return txt(slide,x,y,w,value,size,color,True,'center')
    def card(slide,x,y,w,h,title,body='',color=TEAL,fill=PALE):
        slide.rect(x,y,w,h,fill,color,.8)
        label(slide,x+10,y+11,w-20,title,color,19)
        if body:txt(slide,x+12,y+46,w-24,body,17,INK,align='center')
    def math(slide,latex,x,y,w,h=62,color=INK):
        slide.eq(x,y,w,h,latex,EQUATION_FONT_SIZE,color)

    # 1 — Establish the complete task before showing the calculation details.
    s=add('蓝图设计：把已有证据变成可检验的模型',
          '没有适用的现成模型时，用人体模板、药物资料和少量开发观测完成构建。','一条主线')
    for i,(title,body,col,fill) in enumerate([
        ('输入','人体＋药物资料\n剂量＋开发观测',TEAL,PALE),
        ('构建人体方程','血流连接器官\n计算分布与清除',GREEN,GREEN_PALE),
        ('校准与检查','只调允许的参数\n检查范围冲突',GOLD,GOLD_PALE),
        ('锁定后预测','新给药条件\n曲线＋终点',PURPLE,PURPLE_PALE)]):
        x=50+220*i;card(s,x,166,198,125,title,body,col,fill)
        if i<3:arr(s,[(x+202,228),(x+215,228)])
    label(s,58,324,846,'方法重点：哪些量保留，哪些量能调，什么情况下允许修复',TEAL,21)
    for x,title,col,fill in [(73,'保护可信证据',TEAL,PALE),(365,'约束未知量校准',GREEN,GREEN_PALE),
                             (657,'诊断并受限修复',GOLD,GOLD_PALE)]:
        s.rect(x,377,230,43,fill,None);label(s,x,385,230,title,col,18)
    carry(s,'目标是交付可追溯、可重复验证的模型；人体模板本身不是我们的新发明。')

    # 2 — One capacity relation connects the input evidence to the state equations.
    s=add('人体给结构，药物资料决定怎样分布和清除',
          '人体提供体积与血流；药物资料通过分配公式转成模型所需的参数。','输入与构建')
    pic(s,plots['human'],51,156,121,270)
    card(s,195,153,302,105,'人体模板','器官体积、组成、血流\n15器官×双区＋3血池',TEAL,PALE)
    card(s,195,284,302,123,'药物与给药资料','fu、logP、pKa、分子量、CL\n剂量、输注时长及证据来源',GREEN,GREEN_PALE)
    arr(s,[(502,204),(531,204),(531,278),(555,278)])
    arr(s,[(502,345),(531,345),(531,278)],GREEN)
    s.rect(565,153,344,254,PALE,TEAL,.8)
    label(s,578,166,318,'体积 × 分配 → 等效容量',TEAL,19)
    math(s,r'K=\sum_c V_c k_c',584,212,305,66)
    math(s,r'C_{equiv}=A/K',584,293,305,58)
    txt(s,583,368,310,'A：药量；K：容药的等效容量。',16,MUTED,align='center')
    carry(s,'分配初值来自PK-Sim Standard或Rodgers–Rowland；K不是器官的真实体积。')

    # 3 — The model's actual organ dynamics, not an optimizer derivation.
    s=add('核心方程：血流搬运药物，血管与组织交换',
          '每个器官分血管与组织两区；按血流连接，组成当前的线性微分方程。','人体核心方程')
    card(s,225,153,237,83,'血管区：药量 Av','容量 Kv',TEAL,PALE)
    card(s,652,153,251,83,'组织区：药量 Ae','容量 Ke',GREEN,GREEN_PALE)
    arr(s,[(58,193),(219,193)]);label(s,54,157,158,'血流带入',TEAL,17)
    arr(s,[(468,193),(646,193)],GREEN,both=True);label(s,477,156,160,'交换 J',GREEN,18)
    arr(s,[(345,239),(345,261),(520,261)]);label(s,370,234,159,'血流带出',TEAL,16)
    math(s,r'J=s_{ex}Q_{eff}R_{BP}\left(\frac{A_v}{K_v}-\frac{A_e}{K_e}\right)',92,278,780,72)
    math(s,r'\frac{dA_v}{dt}=Q(C_{in,b}-C_{v,b})-J',67,360,525,68)
    math(s,r'\frac{dA_e}{dt}=J',621,360,267,68)
    carry(s,'Q是血流，RBP是血/浆比，A/K是等效浓度；J是按浓度差计算的有效交换。')

    # 4 — Only the four effects and the actual development objective are needed.
    s=add('用开发观测调四个倍率，让计算曲线贴近实测',
          '解剖体积、血流和采血位置权重保持既定；未知量只能在允许范围内调整。','观测校准')
    for i,(title,body,col,fill) in enumerate([
        ('清除 sCL','调清除强度',TEAL,PALE),('组织 sK','调组织容量',GREEN,GREEN_PALE),
        ('交换 sex','调交换速度',GOLD,GOLD_PALE),('血细胞 sRBC','调血细胞分配',PURPLE,PURPLE_PALE)]):
        card(s,50+i*220,150,198,89,title,body,col,fill)
    for x,t,col in [(75,'试一组倍率',TEAL),(370,'求解浓度曲线',GREEN),(665,'比较开发观测',GOLD)]:
        s.rect(x,267,220,41,PALE,None);label(s,x,274,220,t,col,18)
        if x<665:arr(s,[(x+227,287),(x+287,287)])
    math(s,r'\hat{\mathbf{s}}=\arg\min_{\mathbf{s}\in\mathcal{S}}\left[\frac{1}{n}\sum_{j=1}^{n}\left(\ln\frac{\hat C_p(t_j;\mathbf{s})}{C_{obs}(t_j)}\right)^2+P(\mathbf{s})\right]',52,325,856,85)
    txt(s,82,414,797,'目标 = 逐点浓度误差＋调整惩罚；S是允许范围，P限制过大的调整。',17,INK,align='center')
    carry(s,'只用开发点；两种分配公式分别校准，再按开发损失选择。拟合好不等于参数唯一。')

    # 5 — Keep the decision evidence, omit numerical optimizer mechanics.
    s=add('不是所有参数都能改：保护证据，冲突才修复',
          '先判断参数范围是否可能解释可见观测，再决定是否允许调整范围。','约束与修复')
    s.rect(50,151,418,282,PALE,TEAL,.8)
    label(s,66,164,386,'① 保护合格输入',TEAL,20)
    txt(s,73,210,372,'例如：净肾清除 CL_R 固定，\n只校准剩余的清除部分。',19,INK)
    math(s,r'CL=CL_R+s_{CL}(CL_0-CL_R)',62,285,394,62)
    txt(s,72,371,374,'无肾清除拆分证据时，CL_R = 0。',16,MUTED)
    s.rect(492,151,418,282,GOLD_PALE,GOLD,.8)
    label(s,508,164,386,'② 用守恒上界检查冲突',GOLD,20)
    math(s,r'C_p(t)\leq D_{delivered}(t)\max_i h_i',508,213,386,64)
    txt(s,513,292,375,'孟鲁司特：可见上界约0.029 mg/L，\n开发实测为0.373 mg/L。',17,INK)
    txt(s,513,351,375,'无冲突：直接锁定。\n有冲突：仅合格未实测RBC下界可修复；\n其余情况停止。',17,GOLD)
    carry(s,'合格实测血液分配会阻止自由RBC校准；不为降低误差而修改可信输入。')

    # 6 — Locked parameters still permit target-human and dosing changes.
    s=add('锁定后换输入：计算新条件的曲线和终点',
          '锁定分配公式和四倍率；新人体条件重建系统，新剂量与时长进入给药项。','锁定预测')
    for i,(title,body,col,fill) in enumerate([
        ('锁定','分配公式＋四个倍率',TEAL,PALE),('改变输入','人体＋剂量＋输注时长',GOLD,GOLD_PALE),
        ('得到预测','血浆浓度曲线＋终点',GREEN,GREEN_PALE)]):
        x=50+i*295;card(s,x,149,270,89,title,body,col,fill)
        if i<2:arr(s,[(x+275,192),(x+288,192)],col)
    math(s,r'\frac{d\mathbf{A}}{dt}=M(\hat{\mathbf{s}};\mathcal{H}_{target})\mathbf{A}+\mathbf{b}r_{target}(t)',50,273,555,73)
    math(s,r'\hat C_p=\mathbf{h}_{target}^{\mathsf{T}}\mathbf{A}',70,360,510,60)
    s.rect(632,264,278,164,GREEN_PALE,GREEN,.8)
    label(s,644,277,254,'同一条曲线 → 11类输出',GREEN,18)
    txt(s,649,318,243,'曲线 / AUC / Cmax / Tmax\nClast / 半衰期 / AUC∞\nCL / Vz / MRT / Vss',17,INK,align='center')
    carry(s,'M汇总血流、交换和清除；h将药量转成采血浓度。目标观测只用于评分。')

    # 7 — One real worked example replaces the step-by-step arithmetic pages.
    s=add('头孢吡肟实例：30分钟输注拟合，3分钟输注预测',
          '用5个开发浓度点校准，随后固定参数；目标研究的观测不参与再次拟合。','一个完整实例')
    label(s,54,147,425,'开发：2000 mg / 30 min / 79 kg',TEAL,18)
    label(s,488,147,425,'目标：2000 mg / 3 min / 70 kg¹',GOLD,18)
    pic(s,plots['Cefepime'],48,180,868,264)
    carry(s,f'开发GMFE {c["gmfe"]:.2f} → 目标{c["target_gmfe"]:.2f}；GMFE越接近1越好。¹70 kg是目标人体假设。')

    # 8 — Separate the existing physics from the project's construction policy.
    s=add('我们的贡献：让“怎么建、能改什么”有依据',
          '复用人体模板与分配公式；重点在证据约束、可行性诊断及可追溯交付。','重点与交付')
    for x,title,body,col,fill in [
        (50,'证据保护','记录来源与单位\n保护合格构建输入',TEAL,PALE),
        (345,'受限校准','未知量有明确作用位置\n倍率有规定的调整范围',GREEN,GREEN_PALE),
        (640,'冲突诊断与修复','先检查物理可行性\n只修复符合条件的范围',GOLD,GOLD_PALE)]:
        card(s,x,166,270,145,title,body,col,fill)
    label(s,65,346,830,'交付：模型与参数 ＋ 曲线和终点 ＋ 来源、校准与修复记录',TEAL,21)
    txt(s,74,395,810,'每个终点经过可计算性检查；来源不够或质量不达标时明确标记。',18,INK,align='center')
    carry(s,'当前适用范围：有本药少量开发观测的单次静脉血浆预测；实例不代表所有药物效果。')
    assert len(pages)==8
    return pages


def main():
    suffixes=('.pptx','.pdf','_预览.png')
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
        render(pages,output_dir=work,stem=STEM,title='PBPK蓝图设计：人体模板到预测（8页重点版）',
            required_terms=['人体','微分方程','校准','头孢吡肟','孟鲁司特','浓度'],
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
