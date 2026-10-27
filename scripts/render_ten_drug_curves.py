#!/usr/bin/env python3
"""Plot the frozen optimized development fit and evaluation study for ten drugs.

No simulations, fitting or source edits. --final exports reviewed glyph-outline
PDF and removes intermediate SVG/grayscale/profile artifacts from this figure.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/blueprint_design/clinical_complete_reference_20260928/ten_drug/optimized_run"
OUT = ROOT / "results/figures/experiments/blueprint_ten_drug"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runtime/cache/paper_figures"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, LogFormatterMathtext, FixedLocator, NullFormatter, FuncFormatter
import numpy as np
import pandas as pd
from PIL import Image

WIDTH_MM, HEIGHT_MM = 185., 215.
COLORS = ("#0072B2", "#D55E00")
MARKERS = ("o", "s")
STYLES = ("-", (0, (4, 2.5)))


def path_of(value):
    path = (ROOT / value).resolve(strict=True)
    if not path.is_relative_to(ROOT):
        raise ValueError("Input must stay inside the project")
    return path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path, audit, expected=None):
    path = path_of(path)
    checksum = digest(path)
    if expected is not None and checksum != expected:
        raise ValueError("Frozen source changed: " + str(path))
    audit[str(path.relative_to(ROOT))] = checksum
    return json.loads(path.read_text())


def load_ref(record, audit):
    return load(record["path"], audit, record["sha256"])


def curve_arrays(native, molecular_weight):
    assert native["time_unit"] == "min" and native["unit"] == "umol/l"
    times = np.asarray(native["time_min"], dtype=float) / 60
    values = np.asarray(native["concentration"], dtype=float) * molecular_weight / 1000
    assert len(times) == len(values) and np.all(np.diff(times) > 0)
    assert np.isfinite(times).all() and np.isfinite(values).all() and (values >= 0).all()
    return times, values


def series(native, times, observed, molecular_weight):
    nt, nv = curve_arrays(native, molecular_weight)
    t, o = np.asarray(times, dtype=float), np.asarray(observed, dtype=float)
    assert len(t) == len(o) and np.all(np.diff(t) > 0) and (o > 0).all()
    assert np.isfinite(t).all() and np.isfinite(o).all()
    assert t.min() >= nt.min() and t.max() <= nt.max(), "No extrapolation of predictions"
    p = np.interp(t, nt, nv)
    assert (p > 0).all()
    error = np.log(p / o)
    display_t = np.unique(np.r_[t.min(), nt[(nt >= t.min()) & (nt <= t.max())], t.max()])
    display_p = np.interp(display_t, nt, nv)
    assert (display_p > 0).all()
    return {"time": t, "observed": o, "sampled_prediction": p, "plot_time": display_t,
            "plot_prediction": display_p, "gmfe": float(np.exp(np.mean(np.abs(error)))),
            "ln_rmse": float(np.sqrt(np.mean(error**2))),
            "within_twofold_fraction": float(np.mean(np.abs(error) <= math.log(2)))}


def collect():
    audit, panels, rows = {}, [], []
    summary = load(SOURCE / "summary.json", audit)
    assert summary["status"] == "reported"
    document = load(SOURCE / "comparison_input.json", audit)
    cases = {c["case_id"]: c for c in document["cases"]}
    assert len(cases) == len(summary["protocol"]["cases"]) == 10
    for key in ("pins",):
        for name, checksum in summary["protocol"][key].items():
            assert digest(path_of(name)) == checksum, name
    for locks in (summary["locks"]["inputs"], summary["locks"]["mappings"]):
        for name, checksum in locks.items():
            assert digest(path_of(name)) == checksum, name
    for config in summary["protocol"]["cases"]:
        cid, drug = config["case_id"], config["drug"]
        target = cases[cid]
        dev = load_ref(config["development_source"], audit)
        phys = load(config["physchem"], audit, summary["protocol"]["pins"][config["physchem"]])
        mw = phys["identity"]["molecular_weight_g_mol"]
        selected = summary["selection"][cid]
        mapping_path = path_of(selected["mapping_path"])
        mapping = load(mapping_path, audit, summary["locks"]["mappings"][selected["mapping_path"]])
        assert mapping["eligible"] and mapping["construction_partition_method"] == selected["formula"]
        assert mapping["reference_human"] == dev["human"] and mapping["reference_dosing"] == dev["dosing"]
        assert mapping["training_source"] == config["development_source"]
        assert dev["study_id"] != config["study_id"]
        attempt = mapping_path.parent / f"attempt_{mapping['decision']['evaluation']:02d}"
        trial = load(attempt / "result.json", audit)
        assert trial["status"] == "predicted"
        assert load(attempt / "parameter_audit_result.json", audit)["accepted"]
        native_dev = load(attempt / "prediction_locked.json", audit, trial["prediction_sha256"])
        d = series(native_dev, [v["time_min"]/60 for v in dev["observations"]],
                   [v["concentration_umol_per_l"]*mw/1000 for v in dev["observations"]], mw)
        chosen = mapping["evaluations"][mapping["decision"]["evaluation"]-1]
        np.testing.assert_allclose(d["sampled_prediction"]*1000/mw,
                                   chosen["predicted_at_observed_times"], rtol=1e-10, atol=1e-13)
        assert math.isclose(d["gmfe"], mapping["decision"]["curve_metrics"]["gmfe"], rel_tol=1e-10)
        native_e = load_ref(summary["locks"]["predictions"][cid]["ours"], audit)
        predicted = load_ref(target["predictions"]["ours"], audit)
        construction_path = path_of(summary["results"][cid]["ours"]["prediction_path"]).parent / "construction.json"
        construction = load(construction_path, audit)
        assert construction["mapping_sha256"] == digest(mapping_path)
        assert construction["decision"] == mapping["decision"]
        clinical = target["clinical_reference"]["curve"]
        assert clinical["unit"] == "ng/mL"
        e = series(native_e, clinical["time_h"], np.asarray(clinical["concentration"])/1000, mw)
        q = np.interp(e["time"], predicted["curve"]["time_h"], np.asarray(predicted["curve"]["concentration"])/1000)
        np.testing.assert_allclose(e["sampled_prediction"], q, rtol=1e-11, atol=1e-14)
        assert clinical["matrix"] == "plasma" or (clinical["matrix"] == "serum" and "serum_as_plasma" in target["approximations"])
        metadata = []
        for label, item, human, dosing, study in [
            ("Development", d, dev["human"], dev["dosing"], dev["study_id"]),
            ("Evaluation", e, config["target_request_metadata"]["human"], config["target_request_metadata"]["dosing"], config["study_id"])]:
            metadata.append({"stage": label, "study_id": study, "human_model_inputs": human, "dosing": dosing,
                "observed_points": len(item["time"]), "display_window_h": [float(item["time"].min()), float(item["time"].max())],
                "gmfe": item["gmfe"], "ln_rmse": item["ln_rmse"], "within_twofold_fraction": item["within_twofold_fraction"],
                "dense_prediction_points_displayed": len(item["plot_time"]),
                "observed_time_h": item["time"].tolist(), "observed_mg_l": item["observed"].tolist(),
                "prediction_at_observations_mg_l": item["sampled_prediction"].tolist()})
            for t, value, p in zip(item["time"], item["observed"], item["sampled_prediction"]):
                rows.append({"drug": drug, "stage": label, "time_h": t, "observed_mg_l": value, "predicted_mg_l": p})
        detail = {"case_id": cid, "drug": drug, "formula": selected["formula"], "same_selected_parameters_for_both_studies": True,
            "selected_development_evaluation": mapping["decision"]["evaluation"],
            "selected_parameters": {k:mapping["decision"][k] for k in ("lipophilicity", "clearance_scale")},
            "development_native": str((attempt/"prediction_locked.json").relative_to(ROOT)),
            "evaluation_native": summary["locks"]["predictions"][cid]["ours"],
            "clinical_evaluation_matrix": clinical["matrix"], "clinical_evaluation_statistic": clinical["statistic"],
            "clinical_evaluation_sample_size": clinical.get("sample_size"), "development_population_source": dev.get("population_source"),
            "development_assumptions": dev.get("demographic_assumptions", []), "development_limitations": dev.get("limitations", []),
            "evaluation_assumptions": config["target_request_metadata"].get("demographic_assumptions", []),
            "clinical_reference_quality_notes": target.get("clinical_reference_quality_notes", []),
            "independence": dev.get("independence"), "series": metadata}
        panels.append({"drug": drug, "series": [d,e], "detail": detail})
    return panels, pd.DataFrame(rows), audit


def export_outlined_pdf(svg_path: Path, png_path: Path) -> dict:
    """Preserve vector glyphs, curves and markers when importing into Tectonic."""
    import cairosvg
    import pymupdf
    import xml.etree.ElementTree as ET

    tree = ET.parse(svg_path)
    assert not any(node.tag.rsplit("}", 1)[-1] in {"text", "image", "pattern"}
                   for node in tree.iter())
    pdf_path = svg_path.with_suffix(".pdf")
    cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path))
    with pymupdf.open(pdf_path) as pdf:
        page = pdf[0]
        expected = (WIDTH_MM / 25.4 * 72, HEIGHT_MM / 25.4 * 72)
        actual = (page.rect.width, page.rect.height)
        assert all(abs(a - b) < .001 for a, b in zip(actual, expected))
        assert not page.get_fonts() and not page.get_images() and not page.get_text().strip()
        drawings = page.get_drawings()
        assert drawings and any(drawing.get("dashes", "[] 0") != "[] 0" for drawing in drawings)
        rendered_pixmap = page.get_pixmap(dpi=300, alpha=False)
        rendered = np.asarray(Image.frombytes("RGB", (rendered_pixmap.width, rendered_pixmap.height),
                                             rendered_pixmap.samples))
    with Image.open(png_path) as png:
        reference = np.asarray(png.convert("RGB"))
    assert all(abs(a - b) <= 1 for a, b in zip(rendered.shape[:2], reference.shape[:2]))
    height, width = (min(a, b) for a, b in zip(rendered.shape[:2], reference.shape[:2]))
    diff = np.abs(rendered[:height, :width].astype(float) - reference[:height, :width].astype(float))
    mean_error = float(diff.mean())
    changed_fraction = float(np.mean(diff.max(axis=2) > 50))
    assert mean_error < 6 and changed_fraction < .05, (mean_error, changed_fraction)
    return {"pdf_text_representation": "vector_outlines", "converter": "CairoSVG",
            "pdf_font_resources": 0, "pdf_raster_images": 0,
            "pdf_vector_paths": len(drawings), "prediction_line_style": "development solid; evaluation dashed",
            "pdf_dash_patterns_preserved": True,
            "pdf_page_size_points": list(actual), "pdf_render_comparison_dpi": 300,
            "pdf_vs_png_mean_absolute_channel_error_0_255": mean_error,
            "pdf_vs_png_pixel_fraction_max_channel_error_above_50": changed_fraction,
            "pdf_render_comparison_passed": True}


def condition_text(series_info):
    d, h = series_info["dosing"], series_info["human_model_inputs"]
    return (f"{d['dose_mg']:.5g} mg / {d['infusion_minutes']:g} min",
            f"{h['weight_kg']:g} kg, {h['age_years']:g} y, {h['sex'][0]}")


def log_time_ticks(low, high):
    """Sparse exact labels on the logarithmic time axis, never move data."""
    candidates = np.asarray([factor*10.**power
        for power in range(math.floor(math.log10(low))-1, math.ceil(math.log10(high))+1)
        for factor in (1., 2., 5.)])
    candidates = candidates[(candidates >= low) & (candidates <= high)]
    if len(candidates) > 5:
        candidates = candidates[np.linspace(0, len(candidates)-1, 5).round().astype(int)]
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--skill-dir", type=Path, default=Path("/home/xuyiqian/.codex/skills/scipilot-figure-skill"))
    args = parser.parse_args()
    sys.path.insert(0, str(args.skill_dir/"scripts"))
    from profile_data import profile_data
    from setup_style import setup_style
    from visual_qa import audit_layout
    from check_figure import check_figure
    panels, observed, pins = collect()
    # Data profiling precedes plotting: positive continuous time/concentration,
    # twenty nested study series; points are means/individual observations,
    # not replicate subjects. Use log curves with unconnected clinical markers.
    profile = profile_data(observed, group_cols=["drug", "stage"])
    OUT.mkdir(parents=True, exist_ok=True)
    fonts = Path(matplotlib.get_data_path())/"fonts/ttf"
    if not fonts.resolve().is_relative_to(ROOT):
        raise ValueError("Use project-local Matplotlib font resources")
    for font in fonts.glob("*.ttf"):
        font_manager.fontManager.addfont(str(font))
    setup_style(journal="general", lang="en", use_sciplots=False)
    plt.rcParams.update({"font.family":["DejaVu Sans"], "font.size":7,
        "axes.labelsize":7.5,"xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":7.5,
        "axes.linewidth":.5,"lines.linewidth":.9,"text.color":"#253447","axes.labelcolor":"#253447",
        "axes.edgecolor":"#798391","xtick.color":"#475467","ytick.color":"#475467",
        "svg.fonttype":"path","figure.constrained_layout.use":False,"axes.unicode_minus":False,
        "savefig.bbox":None})
    fig = plt.figure(figsize=(WIDTH_MM/25.4,HEIGHT_MM/25.4),facecolor="white")
    handles=[Line2D([],[],color=c,linestyle=l,marker=m,markersize=3.6,
                       markerfacecolor="white",markeredgewidth=.75,linewidth=.9)
             for c,l,m in zip(COLORS,STYLES,MARKERS)]
    fig.legend(handles,["Development: fit + observations", "Evaluation: prediction + observations"],
        loc="upper left",bbox_to_anchor=(.076,1.005),frameon=False,ncol=2,columnspacing=1.8,handlelength=2.4)
    fig.text(.015,.51,"Concentration (mg/L, log scale)",rotation=90,va="center",fontsize=7.5)
    fig.text(.535,.014,"Time after dose (h, log scale)",ha="center",fontsize=7.5)
    axes=[]
    for k,panel in enumerate(panels):
        row,col=divmod(k,2)
        left=.09+col*.48; top=.955-row*.184; bottom=top-.161
        width=.395
        drug_label = panel['drug'] + ('*' if panel['drug'] == 'Ceftriaxone' else '')
        fig.text(left,top,f"{chr(65+k)}  {drug_label}",fontsize=8.5,fontweight="bold",va="top")
        for i,info in enumerate(panel['detail']['series']):
            dose,_=condition_text(info)
            stage="Dev" if i==0 else "Eval"
            fig.text(left,top-.017-i*.013,f"{stage}  {dose}",va="top",fontsize=7,color=COLORS[i])
            fig.text(left+width,top-.017-i*.013,f"GMFE {panel['series'][i]['gmfe']:.2f}",
                ha="right",va="top",fontsize=7,color=COLORS[i])
        ys=np.concatenate([np.r_[s['observed'],s['plot_prediction']] for s in panel['series']])
        ts=np.concatenate([s['time'] for s in panel['series']])
        assert (ts > 0).all(), "Log time cannot silently omit nonpositive samples"
        logspan=np.log10(max(ys)/min(ys));pad=max(logspan*.08,.085)
        ylim=(10**(np.log10(min(ys))-pad),10**(np.log10(max(ys))+pad))
        tpad=max(.035, np.log10(max(ts)/min(ts))*.04)
        xlim=(10**(np.log10(min(ts))-tpad),10**(np.log10(max(ts))+tpad))
        panel['detail']['display_axis_limits']={'time_h':list(xlim),'concentration_mg_l':list(ylim)}
        ax=fig.add_axes([left,bottom,width,.115]); axes.append(ax)
        for i,item in enumerate(panel['series']):
            ax.plot(item['plot_time'],item['plot_prediction'],color=COLORS[i],linestyle=STYLES[i],zorder=2)
            # Opaque white centers separate symbols from the trajectories.
            # A smaller evaluation square preserves the development circle's
            # outer rim at coincident coordinates, without displacing points.
            points=ax.scatter(item['time'],item['observed'],marker=MARKERS[i],s=16 if i==0 else 9,
                facecolor='white',edgecolor=COLORS[i],linewidth=.65,zorder=3+i)
            assert len(points.get_offsets()) == len(item['time'])
        ax.set_xscale('log');ax.set_yscale('log');ax.set_ylim(ylim);ax.set_xlim(xlim)
        ax.yaxis.set_major_locator(LogLocator(base=10,numticks=4));ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        if logspan < 1.6:
            ax.yaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5)))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_locator(FixedLocator(log_time_ticks(*xlim)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(which='major',length=2.5,width=.5,pad=1.8)
        ax.tick_params(which='minor',length=0)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='both',color='#E7EBEF',linewidth=.45,zorder=0)
    fig.canvas.draw();issues=audit_layout(fig)
    renderer=fig.canvas.get_renderer()
    boxes=[(text.get_text(),text.get_window_extent(renderer)) for text in fig.texts]
    boxes += [('Shared legend',fig.legends[0].get_window_extent(renderer))]
    for i,(label,box) in enumerate(boxes):
        for other,other_box in boxes[i+1:]:
            if box.overlaps(other_box):
                issues.append(('FAIL',f'Figure labels overlap: {label} / {other}'))
    assert not any(level=='FAIL' for level,_ in issues),issues
    texts=[t.get_text() for t in fig.findobj(matplotlib.text.Text)]
    assert not any(re.search(r'[\u3400-\u9fff]',t) for t in texts)
    smallest=min(t.get_fontsize() for t in fig.findobj(matplotlib.text.Text) if t.get_text())
    assert smallest>=7
    basename=OUT/'clinical_curves';fig.savefig(basename.with_suffix('.png'),dpi=300,facecolor='white')
    gray=OUT/'clinical_curves_grayscale.png'
    with Image.open(basename.with_suffix('.png')) as im:im.convert('L').save(gray,dpi=(300,300))
    vector={}
    if args.final:
        fig.savefig(basename.with_suffix('.svg'))
        vector=export_outlined_pdf(basename.with_suffix('.svg'),basename.with_suffix('.png'))
    checks={}
    for path in [basename.with_suffix('.png')]+([basename.with_suffix('.pdf')] if args.final else []):
        file_issues,info=check_figure(str(path),min_dpi=300,target_inches=(WIDTH_MM/25.4,HEIGHT_MM/25.4))
        assert not any(level=='FAIL' for level,_ in file_issues),file_issues
        checks[path.name]={'issues':file_issues,'info':info}
    audit={'source':str(SOURCE.relative_to(ROOT)),'figure_mm':[WIDTH_MM,HEIGHT_MM],
        'figure_language':'English','smallest_text_pt':smallest,'panels':10,'study_axes':10,'study_curves':20,
        'development_observed_points':int((observed.stage=='Development').sum()),
        'evaluation_observed_points':int((observed.stage=='Evaluation').sum()),
        'profile':profile,'layout_audit':issues,'file_checks':checks,'pdf_vector_audit':vector,
        'axis':'log10 time after dose in hours and log10 concentration in mg/L; 3–5 readable time ticks; study series share one axis per drug; limits differ between drugs',
        'layout':'Five rows by two columns; one overlaid development/evaluation panel per drug',
        'print_layout':'Drawn at 185 x 215 mm with >=7 pt text for US Letter portrait, not a scaled-down larger figure',
        'caption_requirements':'214 points (110 development and 104 evaluation); development curves fitted; exploration rather than blinded validation; selected parameters shared; human model attributes and assumptions retained in case metadata; Ceftriaxone serum-as-plasma approximation; both axes logarithmic',
        'overlap_handling':'Logarithmic time expands early sampling; development solid/evaluation dashed thin lines; differently sized open circle/square markers with opaque white centers; no jitter, point omission or data changes',
        'window':'Each prediction line is restricted to the first through last observation of its own study.',
        'interpretation':'Exploratory same-drug cross-study comparison after refinement; development curves are fitted. Not blinded validation or an unseen-drug test.',
        'uncertainty':'No confidence bands or subject-level variability invented; symbols are published group summaries except individual dexmedetomidine evaluation.',
        'serum_approximation':'Ceftriaxone evaluation reports serum, compared with predicted plasma under the frozen serum_as_plasma approximation.',
        'point_pairing':'Each native prediction is interpolated linearly only at observations of the matching study; GMFE=exp(mean(abs(log(P/O)))).',
        'data_checks':['All20 native curves match their study inputs and selected shared mapping','Selected development curves checked against stored sampled optimizer predictions and GMFE','Evaluation curves match frozen report interpolation','All214 original curated observations retained; no extrapolation','Source and prediction checksums verified'],
        'cases':[p['detail'] for p in panels],'input_sha256':pins,
        'plotting_script_sha256':digest(Path(__file__))}
    (OUT/'figure_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False,default=str)+'\n')
    if args.final:
        for path in [gray,basename.with_suffix('.svg'),OUT/'data_profile.json']:
            path.unlink(missing_ok=True)
    print(json.dumps({'figure_mm':[WIDTH_MM,HEIGHT_MM],'observations':len(observed),'layout_issues':issues,
                     'vector_exported':args.final,'gmfe':{p['drug']:[v['gmfe'] for v in p['series']] for p in panels}},indent=2))
    plt.close(fig)


if __name__=='__main__':
    main()
